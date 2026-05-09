#!/usr/bin/env python3
"""
Demo script that runs the inflection detector on sample transcripts
representing known historical inflection points (NVDA AI pivot, LLY GLP-1 obesity pivot,
MU HBM/AI memory, CH Robinson AI automation, XPO margin transformation).

Use this to verify the full pipeline is working before connecting to AlphaSense.
"""
import datetime
import os
import sys

from src.utils.logger import setup_logger, get_logger
from src.utils.cache import TranscriptCache
from src.collectors.base import TranscriptRecord
from src.nlp.pass1_keywords import KeywordMatcher
from src.nlp.pass2_ner import NERExtractor
from src.nlp.pass3_classifier import ZeroShotClassifier
from src.nlp.novelty_detector import NoveltyDetector
from src.scoring.scorer import Scorer
from src.output.excel_writer import write_excel, make_output_path

setup_logger("data/inflection_detector.log", "INFO")
logger = get_logger("inflection_detector.demo")

# ─── Sample Transcript Content ──────────────────────────────────────────────
# These are representative excerpts based on publicly known management commentary
# that preceded major re-ratings of each stock.

SAMPLE_TRANSCRIPTS = [
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "date": "2023-02-22",
        "quarter": "Q4 FY2023",
        "sector": "Information Technology",
        "source": "demo",
        "url": "https://investor.nvidia.com/",
        "content": """
Operator: Welcome to the NVIDIA Fourth Quarter Fiscal Year 2023 earnings conference call.

Jensen Huang - CEO, NVIDIA:
Thank you. NVIDIA is at an extraordinary moment. The transition from traditional computing to accelerated computing and generative AI is happening now, and it is moving fast. We are at a tipping point.

Every data center in the world will be modernized to accelerated computing. The companies that are racing to build generative AI are creating an entirely new class of computing infrastructure. This new use case for GPUs is something we never anticipated at this scale.

Our data center revenue reached an inflection point this quarter. We see demand for our H100 GPU platform accelerating dramatically across cloud service providers, enterprise, and nation-state customers. The addressable market for data center AI infrastructure is approaching a $1 trillion opportunity over the next decade.

The inference workloads for large language models are creating unprecedented GPU demand. Every major cloud provider — Amazon, Microsoft, Google — is deploying our GPU clusters at a scale that was impossible to forecast. This is a platform shift of historic proportions.

Colette Kress - CFO:
Data center revenue was $3.6 billion, up 11% sequentially and up 8% from a year ago. However, we expect data center revenue to reach $4.5 billion next quarter, which would represent meaningful acceleration. The demand environment for AI training and inference workloads has never been stronger.

We now see hyperscaler orders extending beyond 12 months with no signs of slowing. The AI training scaling laws are driving customers to purchase significantly more GPU compute than originally planned. Every dollar of data center GPU revenue drives additional networking, memory, and software revenue.

Jensen Huang - CEO:
Let me be more specific. Six months ago, generative AI and large language models were research projects. Now they are production services used by hundreds of millions of people. This behavioral shift in how AI is deployed is permanent and structural. ChatGPT, Midjourney, and dozens of new enterprise applications are driving this. The demand for inference infrastructure alone — deploying these models to end users — will require GPU compute that dwarfs what we've seen in training.

We are seeing cross-industry adoption that is remarkable. Automotive companies, pharmaceutical companies, financial services firms — every industry is now racing to build AI capabilities. This is not a cyclical dynamic. This is the beginning of a decade-long platform transition.

Analyst Question:
With so much competition coming from AMD, Intel, and Google's TPUs, how confident are you in maintaining your market position?

Jensen Huang:
Our CUDA ecosystem has 4 million developers and 25 years of software investment. This is a moat that takes a decade to replicate. But more importantly, the market is so large that multiple players will succeed. We are focused on the $1 trillion infrastructure opportunity. The transition to AI computing will lift all boats.

Analyst Question:
Can you quantify the AI training versus inference split in your current revenue?

Colette Kress:
We don't break it out precisely, but we can say that inference is growing faster than training as companies move from building models to deploying them. Both segments are growing rapidly.
""",
    },
    {
        "ticker": "LLY",
        "company_name": "Eli Lilly and Company",
        "date": "2022-10-27",
        "quarter": "Q3 2022",
        "sector": "Health Care",
        "source": "demo",
        "url": "https://investor.lilly.com/",
        "content": """
Operator: Welcome to Eli Lilly's third quarter 2022 earnings call.

David Ricks - CEO, Eli Lilly:
We're pleased to report strong third quarter results. But I want to spend time today on something more important than quarterly numbers. We are witnessing an unexpected and potentially transformational development with our GLP-1 platform.

Tirzepatide, which we approved for diabetes as Mounjaro, is demonstrating weight loss in our SURMOUNT clinical program that we frankly did not anticipate at this magnitude. In the obesity trials, we're seeing patients lose 22% of body weight — a level of efficacy that is approaching what's achievable with bariatric surgery.

This is a structural shift in how physicians and patients think about obesity treatment. Obesity is being reclassified from a behavioral issue to a metabolic disease. The regulatory environment is becoming more favorable. CMS is considering coverage for obesity treatment for the first time in Medicare. If that happens, the addressable market for tirzepatide in obesity alone could reach $50 billion annually in the United States.

Anne White - President, Lilly Medicines:
What we're seeing in clinical practice is a consumer behavior change that is unlike anything we've seen before. Patients are proactively requesting these medications. Physicians who never prescribed obesity treatments are now prescribing them. Primary care doctors, cardiologists, endocrinologists — the demand is coming from multiple directions simultaneously.

We believe obesity treatment will become the standard of care within five years. The total addressable market for GLP-1 medications in obesity, diabetes, cardiovascular disease, and emerging indications could exceed $100 billion globally. This is not a niche opportunity. This is a potential transformation of multiple therapeutic areas.

David Ricks:
Let me be direct about what this means for Lilly. We are in the early innings of a decades-long growth opportunity. The manufacturing investments we are making today — billions of dollars in new capacity — reflect our confidence that demand for tirzepatide will far exceed current projections. We have never seen this kind of sustained, inelastic demand for any product in our history.

Analyst Question:
How confident are you in the cardiovascular outcomes data for tirzepatide?

Anne White:
Our SURPASS-CVOT data is enrolling faster than expected because cardiologists see this as a potential breakthrough for their highest-risk patients. Preliminary data suggests not just weight loss but meaningful improvement in cardiac risk markers. If the outcomes data confirms what we're seeing in biomarkers, this changes how cardiologists think about metabolic disease management.

Analyst Question:
What's limiting your ability to ramp manufacturing faster?

Josh Smiley - CFO:
We're investing $3.5 billion in new manufacturing capacity this year and have committed to further expansions. The challenge is that we underestimated demand by a factor of two to three times. We are working 24/7 to expand capacity but we're also not going to sacrifice quality. The demand environment gives us confidence that every vial we produce will be sold.
""",
    },
    {
        "ticker": "MU",
        "company_name": "Micron Technology",
        "date": "2023-09-27",
        "quarter": "Q4 FY2023",
        "sector": "Information Technology",
        "source": "demo",
        "url": "https://investors.micron.com/",
        "content": """
Operator: Welcome to Micron Technology's fiscal fourth quarter 2023 earnings call.

Sanjay Mehrotra - CEO, Micron:
We are at an inflection point for the memory industry, and specifically for Micron. The development of large language models and the AI training paradigm has fundamentally changed the value proposition of high-bandwidth memory.

Let me explain why. Training a large language model — GPT-4, Llama, or any frontier model — requires holding massive amounts of intermediate data in memory during training. The KV cache, which stores key-value pairs for attention mechanisms, scales with model size. A 100-billion parameter model requires terabytes of memory that must be accessed at extraordinary bandwidth. This is creating demand for HBM — high bandwidth memory — that the industry was not prepared for.

Our HBM3E product offers 2.5 times the bandwidth and 50% better power efficiency compared to HBM2E. NVIDIA, AMD, and Intel are all designing their next-generation AI accelerators around HBM3E. This is not a feature upgrade — this is a platform shift where memory bandwidth becomes the primary constraint in AI compute.

Mark Murphy - CFO:
HBM has historically been a niche product for high-performance computing. AI training has transformed it into a high-volume strategic product. Our HBM revenue is on track to reach the hundreds of millions of dollars this fiscal year and we expect it to scale into the billions as AI infrastructure deployment accelerates.

Importantly, HBM average selling prices are 3 to 5 times higher than comparable DRAM capacity. The AI use case is driving a positive mix shift that we expect to be durable for many years.

Sanjay Mehrotra:
There's a second dynamic I want to highlight: inference. When you deploy an AI model — running it for millions of users — you need to store the model weights and KV cache in memory, and you need to do it at low latency. Inference is now a larger workload than training by compute hours. Every inference workload requires substantial memory bandwidth. The transition from AI training to AI deployment at scale is creating a structural increase in memory content per server.

We are seeing new customer segments — enterprise customers building private AI infrastructure — who were not memory buyers at scale before. This cross-industry adoption is just beginning. Healthcare companies, financial firms, manufacturing companies are all building AI capabilities. Each of them requires memory infrastructure. The total addressable market expansion is real and it's happening faster than our original forecast.

Analyst Question:
How should we think about the competitive dynamics in HBM given Samsung and SK Hynix's aggressive expansion?

Sanjay Mehrotra:
The demand for HBM will outstrip total industry supply for at least the next 12 to 18 months. Every HBM wafer that we, Samsung, and SK Hynix can produce will be consumed by AI infrastructure. The question is not about competition — it's about whether the industry can ramp fast enough to meet demand. We are expanding our HBM capacity as fast as we can while maintaining our quality advantage.
""",
    },
    {
        "ticker": "CHRW",
        "company_name": "C.H. Robinson Worldwide",
        "date": "2023-07-26",
        "quarter": "Q2 2023",
        "sector": "Industrials",
        "source": "demo",
        "url": "https://ir.chrobinson.com/",
        "content": """
Operator: Welcome to C.H. Robinson's second quarter 2023 earnings call.

Dave Bozeman - CEO, C.H. Robinson:
Since joining nine months ago, my team and I have conducted a comprehensive assessment of our business. The findings were clear: we have extraordinary technology and data assets that were not being leveraged to their full potential. We are executing a transformation that will structurally change our cost structure and operating model.

Let me be specific. We have deployed AI-enabled automation across our freight matching, pricing, and carrier communication workflows. In our NAST North American Surface Transportation business, we have automated 50% of the load management tasks that previously required human intervention. This is reducing our headcount requirements while improving service quality.

Our operating expense reduction initiative is not a one-time restructuring. It is a fundamental redesign of how we operate. We expect to take out $300 million in costs over 24 months. This is not about cutting corners — it's about using technology to do more with less.

Michael Zechmeister - CFO:
Our operating expense ratio improved by 200 basis points sequentially. This is the early signal of the operating leverage we expect to achieve as we continue deploying automation. The AI-enabled productivity improvements are ahead of our original plan.

We're seeing efficiency gains that compound: as we automate more processes, our people can focus on higher-value customer interactions. This is driving both cost reduction and revenue quality improvement. The gross margin on AI-assisted transactions is 15% higher than on manually processed transactions.

Dave Bozeman:
I want to be clear about the structural nature of these changes. This is not a temporary cost reduction during a freight downturn. The pricing power we are recovering through better data and analytics is permanent. The automation we are deploying removes variable cost regardless of freight market conditions. We expect our operating income margin to reach 30% — a level we have never achieved before — within three years.

Analyst Question:
How should we think about the pace of headcount reduction?

Michael Zechmeister:
We've already reduced headcount by 2,000 people this year and we expect further reductions through the balance of 2023. Importantly, the cost takeouts are occurring faster than volume decline, which means each person we remove represents a permanent efficiency gain, not a temporary cut during a down cycle.

Analyst Question:
What's your confidence level that the AI automation gains are truly structural?

Dave Bozeman:
Every process we've automated, we've studied carefully to ensure we're not sacrificing carrier or customer quality. The results show the opposite — automated processes have lower error rates and faster cycle times. This is not a 12-month science project. We have six-month data confirming the thesis. This is real, it is happening now, and it will fundamentally change our competitive position.
""",
    },
    {
        "ticker": "XPO",
        "company_name": "XPO, Inc.",
        "date": "2023-10-31",
        "quarter": "Q3 2023",
        "sector": "Industrials",
        "source": "demo",
        "url": "https://ir.xpo.com/",
        "content": """
Operator: Welcome to XPO's third quarter 2023 earnings conference call.

Brad Jacobs - CEO, XPO:
I'm delighted with our performance this quarter. But the numbers don't fully capture the transformation underway at XPO. We have fundamentally changed our operating model, and we are at a point where the changes are becoming visible in our financials.

Let me be direct: the LTL business we operate today is structurally different from what it was 18 months ago. We have improved our on-time delivery rate from 89% to 96%. We have reduced our claims ratio by 40%. And we have improved our operating ratio — our cost per dollar of revenue — by 340 basis points.

This is operational transformation through technology and process redesign, not through cutting service. We deployed AI-driven load planning that optimizes freight placement across our 290-terminal network in real time. This reduces empty miles by 12% and improves asset utilization. The cost savings fall directly to the bottom line.

Kyle Wismans - CFO:
Adjusted EBITDA was $302 million, up 47% year over year. Our adjusted EBITDA margin of 16.4% is the highest in XPO's history. We are raising our full-year guidance to reflect the accelerating margin improvement.

The pricing discipline we have established is also holding. Revenue per shipment increased 8% year over year despite a softer freight market. This is pricing power that comes from service quality. Customers pay for reliability, and we are delivering reliability at a level we never achieved before.

Brad Jacobs:
Let me give you a sense of where we're going. The best-in-class LTL carriers operate at 70% to 73% operating ratios — we are at 84% today. Closing that gap represents $400 to $600 million of additional EBITDA annually. We believe we can achieve that within 3 years through continued operational improvement.

The structural changes we've made — technology investment, workforce productivity, network optimization — are permanent. This is not about riding a freight cycle. We have changed how this company operates.

Analyst Question:
Is the pricing power sustainable in a weak freight environment?

Kyle Wismans:
The correlation between our service improvements and customer retention is the data we look at most carefully. Customers who experienced our improved service are renewing contracts at higher rates and higher prices than before. Service quality creates pricing power that transcends freight market cycles. We're seeing it in our yield data and we expect it to continue.
""",
    },
]


def build_transcript_record(sample: dict) -> TranscriptRecord:
    raw_text = sample["content"].strip()
    # Simple speaker/section splitting
    lines = raw_text.split("\n")
    prepared = []
    qa = []
    in_qa = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Analyst Question" in line or "Analyst:" in line:
            in_qa = True
        if in_qa:
            qa.append(line)
        else:
            prepared.append(line)

    return TranscriptRecord(
        ticker=sample["ticker"],
        company_name=sample["company_name"],
        date=sample["date"],
        source=sample["source"],
        quarter=sample["quarter"],
        raw_text=raw_text,
        speakers=[
            {"name": "CEO", "title": "Chief Executive Officer", "role": "CEO"},
            {"name": "CFO", "title": "Chief Financial Officer", "role": "CFO"},
            {"name": "Analyst", "title": "Equity Research Analyst", "role": "Analyst"},
        ],
        sections={"prepared_remarks": prepared, "qa": qa},
        url=sample["url"],
        fetched_at=datetime.datetime.utcnow().isoformat(),
    )


def main():
    import yaml
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    cache = TranscriptCache(config["processing"]["cache_dir"])
    keyword_matcher = KeywordMatcher("config/signal_patterns.yaml")
    ner_extractor = NERExtractor(config["nlp"]["spacy_model"])
    classifier = ZeroShotClassifier(skip=True)  # Skip heavy model in demo
    novelty_detector = NoveltyDetector(default_score=0.75)  # High novelty for demo
    scorer = Scorer(config["scoring"])

    all_signals = []
    sector_map = {s["ticker"]: s["sector"] for s in SAMPLE_TRANSCRIPTS}

    print("\nRunning inflection detector on sample transcripts...\n")

    for sample in SAMPLE_TRANSCRIPTS:
        record = build_transcript_record(sample)
        print(f"Processing {record.ticker} ({record.quarter})...")

        kw_matches = keyword_matcher.match(record)
        if not kw_matches:
            print(f"  No keyword matches — check pattern file")
            continue
        print(f"  Found {len(kw_matches)} keyword matches across {len(set(m.category for m in kw_matches))} categories")

        signal_sentences = list({m.sentence for m in kw_matches if m.sentence})[:15]
        ner_result = ner_extractor.extract(record.raw_text, signal_sentences)

        from src.nlp.novelty_detector import NoveltyResult
        novelty_result = NoveltyResult(score=0.75, prior_count=0, is_cold_start=True)

        signals = scorer.score(record, kw_matches, ner_result, None, novelty_result, sample["sector"])
        print(f"  Generated {len(signals)} signals: {[(s.tier, round(s.score,1), s.inflection_type) for s in signals]}")
        all_signals.extend(signals)

    print(f"\nTotal signals found: {len(all_signals)}")
    t1 = sum(1 for s in all_signals if s.tier == "Tier 1")
    t2 = sum(1 for s in all_signals if s.tier == "Tier 2")
    t3 = sum(1 for s in all_signals if s.tier == "Tier 3")
    print(f"  Tier 1: {t1}  |  Tier 2: {t2}  |  Tier 3: {t3}")

    output_path = make_output_path("data/output/demo_inflection_report_{timestamp}.xlsx")
    write_excel(all_signals, output_path, [])
    print(f"\nOutput written to: {output_path}")
    print("Open the Excel file to review results with color-coded tier scoring.\n")


if __name__ == "__main__":
    main()
