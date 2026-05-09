#!/usr/bin/env python3
"""
Transcript Inflection Point Detector
Scans public company earnings call transcripts to surface material inflection signals.
"""
import argparse
import csv
import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import yaml

from src.utils.logger import setup_logger, get_logger
from src.utils.cache import TranscriptCache
from src.collectors.edgar import EdgarCollector
from src.nlp.pass1_keywords import KeywordMatcher
from src.nlp.pass2_ner import NERExtractor
from src.nlp.pass3_classifier import ZeroShotClassifier
from src.nlp.novelty_detector import NoveltyDetector
from src.scoring.scorer import Scorer, InflectionSignal
from src.output.excel_writer import write_excel, make_output_path

logger = get_logger("inflection_detector")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_companies(csv_path: str = "config/companies.csv", enabled_only: bool = True) -> list[dict]:
    companies = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if enabled_only and row.get("enabled", "true").lower() != "true":
                continue
            if row.get("cik"):
                companies.append(row)
    return companies


def filter_companies(companies: list[dict], tickers: Optional[list[str]]) -> list[dict]:
    if not tickers:
        return companies
    ticker_set = {t.upper() for t in tickers}
    return [c for c in companies if c["ticker"].upper() in ticker_set]


def filter_by_market_cap(companies: list[dict], max_cap_bn: float) -> list[dict]:
    """Keep only companies with market cap ≤ max_cap_bn billion (via yfinance)."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping market cap filter")
        return companies

    kept = []
    for company in companies:
        ticker = company["ticker"].upper()
        try:
            info = yf.Ticker(ticker).fast_info
            cap_bn = (getattr(info, "market_cap", None) or 0) / 1e9
            if cap_bn <= max_cap_bn:
                kept.append(company)
            else:
                logger.info(f"{ticker}: ${cap_bn:.1f}bn market cap — filtered (>{max_cap_bn}bn)")
        except Exception as e:
            logger.debug(f"{ticker}: market cap lookup failed ({e}) — included by default")
            kept.append(company)
    return kept


def init_collectors(config: dict):
    collectors = {}
    src_cfg = config.get("sources", {})

    if src_cfg.get("edgar", {}).get("enabled", True):
        collectors["edgar"] = EdgarCollector(src_cfg.get("edgar", {}))

    if src_cfg.get("seeking_alpha", {}).get("enabled", False):
        from src.collectors.seeking_alpha import SeekingAlphaCollector
        collectors["seeking_alpha"] = SeekingAlphaCollector(src_cfg.get("seeking_alpha", {}))

    if src_cfg.get("alphasense", {}).get("enabled", False):
        from src.collectors.alphasense import AlphaSenseCollector
        collectors["alphasense"] = AlphaSenseCollector(src_cfg.get("alphasense", {}))

    if src_cfg.get("ir_website", {}).get("enabled", False):
        from src.collectors.ir_website import IRWebsiteCollector
        collectors["ir_website"] = IRWebsiteCollector(src_cfg.get("ir_website", {}))

    return collectors


def process_ticker(
    company: dict,
    collectors: dict,
    keyword_matcher: KeywordMatcher,
    ner_extractor: NERExtractor,
    classifier: ZeroShotClassifier,
    novelty_detector: NoveltyDetector,
    scorer: Scorer,
    cache: TranscriptCache,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> tuple[list[InflectionSignal], list[dict]]:
    ticker = company["ticker"].upper()
    company_name = company["company_name"]
    sector = company.get("sector", "")
    ir_url = company.get("ir_url", "")

    log_entries = []
    all_signals = []

    for source_name, collector in collectors.items():
        t0 = time.time()
        status = "ok"
        transcripts_found = 0
        error_msg = ""

        try:
            if source_name == "ir_website":
                records = collector.collect(ticker, company_name, start_date, end_date, cache, ir_url)
            else:
                records = collector.collect(ticker, company_name, start_date, end_date, cache)

            transcripts_found = len(records)

            if not dry_run:
                for record in records:
                    if record.parse_status != "ok" or record.word_count < 500:
                        continue

                    # NLP passes
                    kw_matches = keyword_matcher.match(record)
                    if not kw_matches:
                        continue

                    signal_sentences = list({m.sentence for m in kw_matches if m.sentence})[:20]

                    ner_result = ner_extractor.extract(record.raw_text, signal_sentences)

                    classifier_result = None
                    if signal_sentences:
                        results = classifier.classify_sentences(signal_sentences[:10])
                        if results:
                            classifier_result = results[0]

                    # Novelty detection
                    prior_paths = cache.list_prior_embeddings(ticker, record.date)
                    novelty_result = novelty_detector.compute(signal_sentences, prior_paths)

                    # Save embeddings for future novelty comparisons
                    emb_path = cache.embedding_path(ticker, record.date)
                    if not os.path.exists(emb_path) and signal_sentences:
                        novelty_detector.save_embeddings(signal_sentences, emb_path)

                    signals = scorer.score(record, kw_matches, ner_result, classifier_result, novelty_result, sector)
                    all_signals.extend(signals)

        except Exception as e:
            status = "error"
            error_msg = str(e)
            logger.error(f"{ticker}/{source_name}: {e}")

        log_entries.append({
            "ticker": ticker,
            "company": company_name,
            "source": source_name,
            "status": status,
            "transcripts_found": transcripts_found,
            "processing_time_s": time.time() - t0,
            "error_message": error_msg,
        })

    return all_signals, log_entries


def parse_args():
    parser = argparse.ArgumentParser(description="Transcript Inflection Point Detector")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (default: all enabled in companies.csv)")
    parser.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD (overrides config)")
    parser.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD (overrides config)")
    parser.add_argument("--sources", type=str, default="", help="Comma-separated sources to enable")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--dry-run", action="store_true", help="Collect transcripts only, skip NLP")
    parser.add_argument("--output", type=str, default="", help="Override output file path")
    parser.add_argument("--min-tier", type=int, default=3, choices=[1, 2, 3], help="Minimum tier to include in output")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0=use config)")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    parser.add_argument("--skip-zero-shot", action="store_true", help="Skip zero-shot classifier (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.start:
        config["date_range"]["start"] = args.start
    if args.end:
        config["date_range"]["end"] = args.end
    if args.skip_zero_shot:
        config.setdefault("nlp", {})["skip_zero_shot"] = True

    if args.sources:
        enabled_sources = {s.strip() for s in args.sources.split(",")}
        for src in config.get("sources", {}):
            config["sources"][src]["enabled"] = src in enabled_sources

    proc_cfg = config.get("processing", {})
    log_cfg = proc_cfg.get("log_level", "INFO")
    log_file = proc_cfg.get("log_file", "data/inflection_detector.log")
    setup_logger(log_file, log_cfg)

    start_date = config["date_range"]["start"]
    end_date = config["date_range"]["end"]
    nlp_cfg = config.get("nlp", {})

    logger.info(f"Starting scan: {start_date} to {end_date}")
    logger.info(f"Loading companies from config/companies.csv")

    companies = load_companies()
    if args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(",")]
        companies = filter_companies(companies, ticker_list)

    # Market cap filter (skip if specific tickers were given)
    if not args.tickers:
        max_cap = config.get("output", {}).get("max_market_cap_bn")
        if max_cap:
            logger.info(f"Filtering to companies with market cap ≤ ${max_cap}bn...")
            companies = filter_by_market_cap(companies, float(max_cap))

    logger.info(f"Processing {len(companies)} companies")

    os.makedirs("data/cache", exist_ok=True)
    cache = TranscriptCache(proc_cfg.get("cache_dir", "data/cache"))

    # Initialize collectors (expensive — do once)
    collectors = init_collectors(config)
    logger.info(f"Active sources: {list(collectors.keys())}")

    # Initialize NLP models (expensive — do once in main thread)
    logger.info("Loading NLP models...")
    keyword_matcher = KeywordMatcher("config/signal_patterns.yaml")
    ner_extractor = NERExtractor(nlp_cfg.get("spacy_model", "en_core_web_sm"))
    classifier = ZeroShotClassifier(
        model_name=nlp_cfg.get("zero_shot_model", "facebook/bart-large-mnli"),
        threshold=nlp_cfg.get("zero_shot_threshold", 0.45),
        skip=nlp_cfg.get("skip_zero_shot", False),
    )
    novelty_detector = NoveltyDetector(
        model_name=nlp_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        default_score=nlp_cfg.get("novelty_default_score", 0.65),
    )
    scorer = Scorer(config.get("scoring", {}))
    logger.info("NLP models ready")

    all_signals: list[InflectionSignal] = []
    all_log_entries: list[dict] = []

    max_workers = args.workers or proc_cfg.get("max_workers", 4)
    # Use 1 worker if models aren't thread-safe or in dry-run
    if args.dry_run or max_workers == 1:
        for company in companies:
            logger.info(f"Processing {company['ticker']} ({company['company_name']})")
            signals, log_entries = process_ticker(
                company, collectors, keyword_matcher, ner_extractor,
                classifier, novelty_detector, scorer, cache,
                start_date, end_date, args.dry_run,
            )
            all_signals.extend(signals)
            all_log_entries.extend(log_entries)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_ticker, company, collectors, keyword_matcher, ner_extractor,
                    classifier, novelty_detector, scorer, cache,
                    start_date, end_date, args.dry_run,
                ): company["ticker"]
                for company in companies
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    signals, log_entries = future.result()
                    all_signals.extend(signals)
                    all_log_entries.extend(log_entries)
                    logger.info(f"{ticker}: {len(signals)} signals found")
                except Exception as e:
                    logger.error(f"{ticker}: failed — {e}")

    # Filter by tier
    tier_map = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Filtered": 4}
    filtered_signals = [s for s in all_signals if tier_map.get(s.tier, 4) <= args.min_tier]

    # Write output
    output_tmpl = args.output or config.get("output", {}).get("excel_path", "data/output/inflection_report_{timestamp}.xlsx")
    output_path = make_output_path(output_tmpl) if "{timestamp}" in output_tmpl else output_tmpl

    logger.info(f"Writing {len(filtered_signals)} signals to {output_path}")
    write_excel(filtered_signals, output_path, all_log_entries)

    # Summary
    t1 = sum(1 for s in filtered_signals if s.tier == "Tier 1")
    t2 = sum(1 for s in filtered_signals if s.tier == "Tier 2")
    t3 = sum(1 for s in filtered_signals if s.tier == "Tier 3")
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE")
    print(f"  Companies processed: {len(companies)}")
    print(f"  Tier 1 signals:      {t1}")
    print(f"  Tier 2 signals:      {t2}")
    print(f"  Tier 3 signals:      {t3}")
    print(f"  Output:              {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
