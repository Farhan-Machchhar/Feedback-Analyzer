"""
Executive Operational Brief Synthesis Engine.

Aggregates batch customer feedback results into a formal executive memo:
- High-level executive summary
- Top 3 operational bottlenecks
- Root-cause breakdown
- Prioritized managerial action plan
"""
from typing import List, Dict, Any
from collections import Counter
from pydantic import BaseModel, Field

from core.config import settings, logger
from services.analyzer import analyzer


class ExecutiveBrief(BaseModel):
    title: str = "Customer Operations & Sentiment Executive Briefing"
    executive_summary: str = Field(description="High-level overview of overall customer sentiment and operational health")
    critical_risk_alerts: List[str] = Field(description="Safety, legal, or extreme urgency concerns requiring immediate action")
    top_operational_bottlenecks: List[str] = Field(description="Top 3 recurring friction points for customers")
    recommended_action_plan: List[str] = Field(description="Concrete, prioritized corrective initiatives for management")
    sentiment_health_score: str = Field(description="Overall operational grade (e.g., 'Grade: B - Solid food quality, vulnerable delivery logistics')")


class ExecutiveBriefGenerator:
    """Generates managerial executive summaries across batch feedback datasets."""

    @staticmethod
    def generate_brief(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "markdown": "### Executive Brief\n\nNo reviews provided for synthesis.",
                "data": None,
            }

        total = len(results)
        scores = [r.get("overall_score") or r.get("score") or 3 for r in results]
        avg_score = round(sum(scores) / total, 1) if total > 0 else 0.0

        pos_count = sum(1 for r in results if (r.get("overall_label") or r.get("label")) == "positive")
        neg_count = sum(1 for r in results if (r.get("overall_label") or r.get("label")) == "negative")
        pos_pct = round((pos_count / total * 100)) if total > 0 else 0

        # Critical urgency alerts
        critical_alerts = [
            r for r in results
            if (r.get("urgency") or "").lower() in ["high", "critical"]
        ]

        # Themes
        themes = [r.get("theme", "general") for r in results if (r.get("overall_label") or r.get("label")) == "negative"]
        top_negative_themes = [t for t, _ in Counter(themes).most_common(3)] if themes else ["None observed"]

        # Root causes collected
        root_causes = [r.get("root_cause") for r in results if r.get("root_cause")]
        suggested_actions = [r.get("suggested_action") for r in results if r.get("suggested_action")]

        # If live Gemini is available, run generative synthesis
        if analyzer.is_live_llm_ready():
            try:
                sample_texts = [f"- [{r.get('overall_label')} / {r.get('theme')}] {r.get('text') or r.get('review')}" for r in results[:25]]
                prompt = (
                    "You are a VP of Customer Experience and Operations for a retail and dining brand.\n"
                    "Synthesize the following customer reviews into a formal Executive Operational Briefing.\n\n"
                    f"Dataset Stats: Total: {total}, Average Rating: {avg_score}/5, Positive: {pos_pct}%, Negative: {neg_count}.\n"
                    f"Negative Themes: {', '.join(top_negative_themes)}.\n"
                    f"Sample Reviews:\n" + "\n".join(sample_texts) + "\n\n"
                    "Provide a clear, authoritative briefing covering:\n"
                    "1. Executive Summary\n"
                    "2. Critical Safety/Legal Risks (if any)\n"
                    "3. Top 3 Operational Bottlenecks\n"
                    "4. Prioritized Action Plan for General Managers"
                )
                from google.genai import types
                response = analyzer.client.models.generate_content(
                    model=settings.MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExecutiveBrief,
                        temperature=0.2,
                    ),
                )
                brief_parsed: ExecutiveBrief = response.parsed
                md = (
                    f"## 📋 {brief_parsed.title}\n\n"
                    f"**Operational Health**: {brief_parsed.sentiment_health_score}\n\n"
                    f"### 🔍 Executive Summary\n{brief_parsed.executive_summary}\n\n"
                    f"### 🚨 Critical Risk Alerts\n" + "\n".join([f"- ⚠️ {a}" for a in brief_parsed.critical_risk_alerts or ["None identified"]]) + "\n\n"
                    f"### 🚧 Top 3 Operational Bottlenecks\n" + "\n".join([f"1. **{b}**" for b in brief_parsed.top_operational_bottlenecks]) + "\n\n"
                    f"### 🎯 Strategic Action Plan\n" + "\n".join([f"- [ ] {act}" for act in brief_parsed.recommended_action_plan])
                )
                return {"markdown": md, "data": brief_parsed.model_dump()}
            except Exception as e:
                logger.warning(f"Generative brief failed: {e}. Using deterministic synthesis.")

        # Deterministic Synthesis (Offline / Fallback)
        health_grade = (
            "Grade: A (Excellent operational execution)" if avg_score >= 4.2
            else ("Grade: B (Solid core offering, friction in delivery/support)" if avg_score >= 3.2
            else "Grade: C- (Urgent operational turnaround required)")
        )

        critical_bullets = [
            f"**{r.get('theme', 'Urgent').title()} Issue**: \"{r.get('text') or r.get('review')}\""
            for r in critical_alerts[:3]
        ] or ["No food safety or legal threats flagged in this batch."]

        bottlenecks = [
            f"**{theme.replace('_', ' ').title()} Friction**: Dominates negative feedback ({themes.count(theme)} occurrences)."
            for theme in top_negative_themes[:3]
        ]

        action_plan = list(dict.fromkeys(suggested_actions))[:4] or [
            "Review customer communication during peak delivery delays.",
            "Verify kitchen handoff temperature logs.",
            "Conduct weekly frontline service empathy coaching.",
        ]

        md = (
            f"## 📋 Executive Operational Briefing\n\n"
            f"**Operational Health**: `{health_grade}`\n\n"
            f"### 🔍 Executive Summary\n"
            f"Analyzed **{total} customer reviews** with an overall average rating of **{avg_score} / 5.0** ({pos_pct}% positive). "
            f"The primary source of customer friction centers around **{', '.join([t.replace('_', ' ') for t in top_negative_themes])}**.\n\n"
            f"### 🚨 Critical Risk Alerts\n" + "\n".join([f"- ⚠️ {b}" for b in critical_bullets]) + "\n\n"
            f"### 🚧 Top Operational Bottlenecks\n" + "\n".join([f"1. {b}" for b in bottlenecks]) + "\n\n"
            f"### 🎯 Recommended Strategic Action Plan\n" + "\n".join([f"- [ ] {act}" for act in action_plan])
        )

        return {
            "markdown": md,
            "data": {
                "title": "Customer Operations Executive Briefing",
                "sentiment_health_score": health_grade,
                "executive_summary": f"Analyzed {total} reviews with avg score {avg_score}/5.0.",
                "critical_risk_alerts": critical_bullets,
                "top_operational_bottlenecks": bottlenecks,
                "recommended_action_plan": action_plan,
            }
        }


brief_generator = ExecutiveBriefGenerator()

