"""One-off script used to generate the sample PDF documents in data/.
Not part of the RAG pipeline itself -- just a convenience for creating
realistic test fixtures. Requires `fpdf2` (pip install fpdf2).
"""

from fpdf import FPDF


def make_pdf(filename: str, title: str, paragraphs: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for para in paragraphs:
        pdf.multi_cell(0, 6, para)
        pdf.ln(3)
    pdf.output(filename)
    print(f"wrote {filename}")


financial_report = [
    "Section 1: Fiscal Year Overview\n"
    "Nimbus Analytics closed fiscal year 2025 with total revenue of $42.3 million, "
    "representing a 38% year-over-year increase from $30.6 million in fiscal year 2024. "
    "Annual recurring revenue (ARR) reached $47.1 million by the end of Q4, driven primarily "
    "by expansion within the Enterprise tier customer segment.",

    "Section 2: Revenue Breakdown by Segment\n"
    "The Starter tier contributed $6.2 million in revenue (14.6% of total), the Professional "
    "tier contributed $18.9 million (44.7%), and the Enterprise tier contributed $17.2 million "
    "(40.7%). Enterprise tier revenue grew 61% year-over-year, the fastest-growing segment.",

    "Section 3: Operating Expenses\n"
    "Total operating expenses for fiscal year 2025 were $38.7 million, consisting of $19.4 million "
    "in research and development, $11.2 million in sales and marketing, and $8.1 million in general "
    "and administrative costs. Research and development spending increased 45% year-over-year as the "
    "company invested heavily in the StreamSight v3.2 release.",

    "Section 4: Profitability\n"
    "Nimbus Analytics reported an operating income of $3.6 million for fiscal year 2025, marking the "
    "company's first profitable fiscal year since founding. Net income after taxes was $2.4 million, "
    "compared to a net loss of $1.8 million in fiscal year 2024.",

    "Section 5: Customer Metrics\n"
    "The company ended fiscal year 2025 with 1,840 paying customers, up from 1,290 the prior year. "
    "Net revenue retention (NRR) was 118%, and the customer churn rate was 4.2% annually, down from "
    "6.1% in fiscal year 2024. The average contract value (ACV) for Enterprise tier customers was $94,000.",

    "Section 6: Outlook for Fiscal Year 2026\n"
    "Management projects fiscal year 2026 revenue in the range of $56 million to $60 million, representing "
    "32% to 42% growth. This guidance assumes continued expansion in the Enterprise segment and the planned "
    "Q3 launch of StreamSight v4.0, which includes cross-workspace dashboard sharing.",
]

onboarding_guide = [
    "Welcome to Nimbus Analytics! This guide will walk you through your first two weeks as a new "
    "engineering hire.",

    "Day 1: Setup\n"
    "On your first day, IT will provide you with a company laptop pre-configured with the standard "
    "engineering toolchain, including VS Code, Docker, and the Nimbus internal CLI. You will receive "
    "credentials for GitHub, Slack, and the AWS sandbox account. Multi-factor authentication must be "
    "configured on all accounts before you can access any production systems, per company security policy.",

    "Week 1: Onboarding Buddy and Codebase Orientation\n"
    "Every new engineer is paired with an onboarding buddy, an experienced team member who is available "
    "for questions during your first 30 days. During week one, you'll complete the 'StreamSight Architecture' "
    "training module and set up the local development environment following the README in the main "
    "monorepo. You are expected to submit your first pull request, typically a small documentation fix or "
    "test addition, by the end of week one.",

    "Week 2: First Feature Ticket\n"
    "In week two, your manager will assign your first real feature ticket, scoped to be completable within "
    "3-5 days. Code review turnaround time at Nimbus Analytics is expected to be within 24 hours on business "
    "days. All code must pass the CI pipeline, which includes unit tests, linting, and a security scan, "
    "before it can be merged to the main branch.",

    "Engineering Culture\n"
    "Nimbus Analytics follows a blameless post-mortem culture for production incidents. Engineers are "
    "encouraged to write RFCs (Request for Comments) for any architectural change affecting more than one "
    "service. RFCs are reviewed in the weekly Architecture Review meeting held every Wednesday at 2:00 PM.",

    "30-60-90 Day Check-ins\n"
    "Your manager will schedule formal check-in meetings at the 30, 60, and 90 day marks to discuss your "
    "progress, gather feedback on the onboarding process, and set goals for the next period. Your first "
    "formal performance review will take place in the standard June or December review cycle, whichever "
    "comes first after your 90-day mark.",
]

make_pdf("/home/claude/rag-project/data/financial_report_fy2025.pdf", "Nimbus Analytics - FY2025 Financial Report", financial_report)
make_pdf("/home/claude/rag-project/data/engineering_onboarding_guide.pdf", "Nimbus Analytics - Engineering Onboarding Guide", onboarding_guide)
