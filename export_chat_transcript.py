from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
import markdown

SOURCE_JSONL = Path("/Users/shubhdeepdas/.gemini/antigravity/brain/beef5488-f09d-489e-b678-c2e13f19b238/.system_generated/logs/transcript_full.jsonl")
DESKTOP_DIR = Path.home() / "Desktop"
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MD_DESKTOP = DESKTOP_DIR / "Sickle_Cell_Research_Conversation_Transcript.md"
OUTPUT_PDF_DESKTOP = DESKTOP_DIR / "Sickle_Cell_Research_Conversation_Transcript.pdf"
OUTPUT_MD_REPO = DOCS_DIR / "Sickle_Cell_Research_Conversation_Transcript.md"
OUTPUT_PDF_REPO = DOCS_DIR / "Sickle_Cell_Research_Conversation_Transcript.pdf"
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def clean_user_text(raw_text: str) -> str:
    text = raw_text
    if "<USER_REQUEST>" in text:
        text = text.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0]
    # Remove metadata tags
    text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", text, flags=re.DOTALL)
    # Remove context summary if present
    if "<CONTEXT_SUMMARY>" in text:
        text = text.split("</CONTEXT_SUMMARY>")[-1]
    return text.strip()


def parse_transcript(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        steps = [json.loads(line) for line in f]

    turns = []
    current_user = None
    current_replies = []

    for s in steps:
        stype = s.get("type")
        if stype == "USER_INPUT":
            if current_user is not None:
                turns.append({"user": current_user, "replies": current_replies})
            current_user = s
            current_replies = []
        elif stype in ("PLANNER_RESPONSE", "GENERIC"):
            content = s.get("content", "")
            if content and content.strip():
                # Filter out pure command logs
                trimmed = content.strip()
                if not (trimmed.startswith("Created At:") or trimmed.startswith("Tool is running") or trimmed.startswith("Command exited with code")):
                    current_replies.append({"timestamp": s.get("created_at", ""), "text": trimmed})

    if current_user is not None:
        turns.append({"user": current_user, "replies": current_replies})

    return turns


def generate_markdown(turns: list[dict]) -> str:
    md_lines = [
        "# Complete Research Dialogue Transcript: Sickle-Cell Smartphone Microscopy & Shortcut Reliability Audit",
        "",
        "> **Project**: Smartphone Microscopy Sickle-Cell Disease Classification & Shortcut Reliability Audit  ",
        "> **Repository**: `https://github.com/Shubhdeepdsa/sickel_cell_ssl`  ",
        "> **Lead Researcher**: Shubhdeep Das  ",
        "> **AI Assistant**: Google Antigravity (Advanced Agentic Pair Programmer)  ",
        f"> **Generated Date**: {datetime.now().strftime('%B %d, %Y')}  ",
        "> **Source Artifact**: `transcript_full.jsonl` (Conversation ID: `beef5488-f09d-489e-b678-c2e13f19b238`)  ",
        "",
        "---",
        "",
        "## Executive Overview & Transcript Contents",
        "",
        "This document records the complete, unedited technical dialogues, peer-review iterations, model training decisions, and experimental discoveries spanning the entire project lifecycle—from data acquisition and self-supervised model training through the multi-stage artifact audit and peer-review revisions.",
        "",
        "### Table of Conversation Turns",
        "",
    ]

    # Generate TOC
    for idx, t in enumerate(turns):
        u_raw = t["user"].get("content", "")
        u_clean = clean_user_text(u_raw)
        title_summary = u_clean.split("\n")[0][:75].strip()
        if not title_summary:
            title_summary = f"Turn {idx + 1}"
        anchor = f"turn-{idx + 1}"
        md_lines.append(f"{idx + 1}. [**Turn {idx + 1}**: {title_summary}](#{anchor})")

    md_lines.extend(["", "---", ""])

    # Render each turn
    for idx, t in enumerate(turns):
        turn_num = idx + 1
        u_raw = t["user"].get("content", "")
        u_clean = clean_user_text(u_raw)
        ts_user = t["user"].get("created_at", "")
        if ts_user:
            try:
                dt = datetime.fromisoformat(ts_user.replace("Z", "+00:00"))
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                ts_str = ts_user
        else:
            ts_str = "N/A"

        first_line = u_clean.split("\n")[0][:70].strip()
        md_lines.append(f"<a id=\"turn-{turn_num}\"></a>")
        md_lines.append(f"## Turn {turn_num}: {first_line}")
        md_lines.append(f"*Timestamp: {ts_str}*")
        md_lines.append("")
        md_lines.append("### 👤 User Request")
        md_lines.append("")
        md_lines.append(u_clean)
        md_lines.append("")

        replies = t["replies"]
        md_lines.append("### 🤖 Antigravity (AI Assistant) Response")
        md_lines.append("")
        if not replies:
            md_lines.append("*(Command launched as an asynchronous background execution task; results reported in subsequent turn.)*")
            md_lines.append("")
        else:
            # If multiple replies (e.g. intermediate updates), join them
            if len(replies) == 1:
                md_lines.append(replies[0]["text"])
            else:
                # Include intermediate notes or the final response
                for r_idx, r in enumerate(replies):
                    if r_idx < len(replies) - 1:
                        md_lines.append(f"#### Progress Update {r_idx + 1}")
                        md_lines.append(r["text"])
                        md_lines.append("")
                    else:
                        if len(replies) > 1:
                            md_lines.append("#### Final Consolidating Response")
                        md_lines.append(r["text"])
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    return "\n".join(md_lines)


def generate_html(md_text: str) -> str:
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"]
    )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sickle-Cell Research Transcript - Shubhdeep Das & Antigravity</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {{
      content: counter(page);
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1f2328;
    background-color: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 30px 20px;
  }}
  h1 {{
    font-size: 22pt;
    color: #0969da;
    border-bottom: 2px solid #d0d7de;
    padding-bottom: 8px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 15pt;
    color: #1f2328;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 6px;
    margin-top: 28px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 12pt;
    color: #24292f;
    margin-top: 18px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 11pt;
    color: #57606a;
    margin-top: 14px;
    page-break-after: avoid;
  }}
  blockquote {{
    border-left: 4px solid #0969da;
    padding: 6px 14px;
    color: #57606a;
    background-color: #f6f8fa;
    margin: 12px 0;
    font-size: 10pt;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #d0d7de;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{
    background-color: #f6f8fa;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #fbfcfd;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 9.5pt;
    background-color: #eff1f3;
    padding: 2px 4px;
    border-radius: 4px;
  }}
  pre {{
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.45;
    page-break-inside: avoid;
  }}
  pre code {{
    background: none;
    padding: 0;
  }}
  hr {{
    height: 1px;
    background-color: #d0d7de;
    border: none;
    margin: 28px 0;
  }}
  a {{
    color: #0969da;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .user-box {{
    background-color: #f0f7ff;
    border-left: 4px solid #0969da;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 14px 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    return html_template


def main() -> None:
    print(f"Parsing raw JSONL transcript from: {SOURCE_JSONL}")
    turns = parse_transcript(SOURCE_JSONL)
    print(f"Successfully extracted {len(turns)} dialogue turns.")

    print("Generating comprehensive Markdown document...")
    md_content = generate_markdown(turns)
    OUTPUT_MD_DESKTOP.write_text(md_content, encoding="utf-8")
    OUTPUT_MD_REPO.write_text(md_content, encoding="utf-8")
    print(f"  [OK] Saved Markdown to: {OUTPUT_MD_DESKTOP} ({len(md_content)} chars)")
    print(f"  [OK] Saved Markdown to: {OUTPUT_MD_REPO}")

    print("Rendering HTML with modern typography...")
    html_content = generate_html(md_content)
    tmp_html = Path("/tmp/transcript_render.html")
    tmp_html.write_text(html_content, encoding="utf-8")

    print(f"Converting HTML to PDF via Headless Chrome ({CHROME_BIN})...")
    cmd = [
        str(CHROME_BIN),
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={OUTPUT_PDF_DESKTOP}",
        str(tmp_html)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and OUTPUT_PDF_DESKTOP.exists():
        pdf_size_mb = OUTPUT_PDF_DESKTOP.stat().st_size / (1024 * 1024)
        print(f"  [OK] Successfully created Desktop PDF: {OUTPUT_PDF_DESKTOP} ({pdf_size_mb:.2f} MB)")
        # Also copy to repo docs
        import shutil
        shutil.copy2(OUTPUT_PDF_DESKTOP, OUTPUT_PDF_REPO)
        print(f"  [OK] Successfully copied PDF to repo docs: {OUTPUT_PDF_REPO}")
    else:
        print(f"ERROR generating PDF: returncode={res.returncode}")
        print(res.stderr)


if __name__ == "__main__":
    main()
