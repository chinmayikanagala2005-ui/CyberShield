from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from collections import Counter
from html import escape
import joblib, json, os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import load_vectorizer
from src.data_utils import load_json_dataset, simple_clean_text

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = FastAPI(title="Cyberbullying Detection System")

# ----------------- ML Model -----------------
clf = joblib.load("models/model.joblib")
vec = load_vectorizer()

# ----------------- Storage Files -----------------
REPORT_FILE = "reported_messages.json"
BLOCK_FILE = "blocked_users.json"
OFFENSE_FILE = "offense_counts.json"
DATASET_FILE = "data/combined_training_dataset.json"
BLOCK_THRESHOLD = 3


# ----------------- Pydantic Models -----------------
class TextIn(BaseModel):
    username: str
    text: str


# ----------------- Helper Functions -----------------
def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_report(text: str, username: str = "", confidence: float = 0, bullying_terms=None):
    data = _read_json(REPORT_FILE, [])
    data.append({
        "username": username,
        "message": text,
        "confidence": confidence,
        "bullying_terms": bullying_terms or [],
    })
    _write_json(REPORT_FILE, data)


def save_to_blocklist(username: str):
    users = _read_json(BLOCK_FILE, [])
    if username not in users:
        users.append(username)
    _write_json(BLOCK_FILE, users)


def is_blocked(username: str) -> bool:
    users = _read_json(BLOCK_FILE, [])
    return username in users


def get_offense_counts() -> dict:
    return _read_json(OFFENSE_FILE, {})


def add_offense(username: str) -> int:
    """Increase offense count and return updated count."""
    offenses = get_offense_counts()
    current = offenses.get(username, 0) + 1
    offenses[username] = current
    _write_json(OFFENSE_FILE, offenses)
    return current


def identify_bullying_terms(text: str, limit: int = 8) -> list:
    clean = simple_clean_text(text)
    if not clean or not hasattr(clf, "coef_"):
        return []

    X = vec.transform([clean])
    feature_names = vec.get_feature_names_out()
    coefficients = clf.coef_[0]
    term_scores = []

    for feature_index in X.nonzero()[1]:
        score = float(X[0, feature_index] * coefficients[feature_index])
        if score > 0:
            term_scores.append((feature_names[feature_index], score))

    term_scores.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return [
        {"term": term, "score": round(score, 4), "source": "model"}
        for term, score in term_scores[:limit]
    ]


def get_admin_data() -> dict:
    reports = _read_json(REPORT_FILE, [])
    blocked = _read_json(BLOCK_FILE, [])
    offenses = get_offense_counts()
    warning_users = {
        username: count
        for username, count in offenses.items()
        if username not in blocked and count < BLOCK_THRESHOLD
    }
    top_words = Counter()

    for report in reports:
        message = report.get("message", "")
        for word in simple_clean_text(message).split():
            if len(word) > 2:
                top_words[word] += 1

    total_offenses = sum(offenses.values())

    return {
        "reports": reports,
        "blocked": blocked,
        "offenses": offenses,
        "warning_users": warning_users,
        "top_words": top_words.most_common(6),
        "summary": {
            "total_reported_messages": len(reports),
            "total_blocked_users": len(blocked),
            "total_offense_events": total_offenses,
            "users_with_warnings": len(warning_users),
        },
    }


def get_model_performance(data_path=DATASET_FILE) -> dict:
    df = load_json_dataset(data_path)
    df = df[df["label"].notnull()]

    if df.empty:
        return {"available": False, "message": "No labeled evaluation data available."}

    df["label"] = df["label"].astype(int)
    df["clean"] = df["text"].apply(simple_clean_text)

    X = vec.transform(df["clean"])
    y_true = df["label"]
    y_pred = clf.predict(X)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "available": True,
        "dataset_size": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "safe": report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}),
        "bullying": report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}),
        "confusion_matrix": matrix,
    }


def _format_metric(value) -> str:
    return f"{value * 100:.1f}%"


def _render_stat_card(label: str, value, note: str) -> str:
    return f"""
    <article class="stat-card">
        <span>{escape(label)}</span>
        <strong>{escape(str(value))}</strong>
        <small>{escape(note)}</small>
    </article>
    """


def _render_status_pill(count: int, blocked: bool) -> str:
    if blocked:
        return '<span class="pill danger">Blocked</span>'
    if count >= 2:
        return '<span class="pill warning">High Risk</span>'
    return '<span class="pill muted">Warning</span>'


def _render_admin_page() -> str:
    data = get_admin_data()
    summary = data["summary"]
    reports = data["reports"]
    blocked = set(data["blocked"])
    offenses = data["offenses"]
    top_words = data["top_words"]
    performance = get_model_performance()
    max_offense = max(offenses.values(), default=1)
    blocked_count = summary["total_blocked_users"]
    warning_count = summary["users_with_warnings"]
    reports_count = summary["total_reported_messages"]
    pie_total = max(blocked_count + warning_count + reports_count, 1)
    reported_pct = round((reports_count / pie_total) * 100, 1)
    blocked_pct = round((blocked_count / pie_total) * 100, 1)
    warning_pct = round((warning_count / pie_total) * 100, 1)
    blocked_slice = (blocked_count / pie_total) * 100
    warning_slice = (warning_count / pie_total) * 100
    user_rows = []

    for username, count in sorted(offenses.items(), key=lambda item: (-item[1], item[0])):
        width = max(8, int((count / max_offense) * 100))
        user_rows.append(f"""
            <tr>
                <td>{escape(username)}</td>
                <td>{count}/{BLOCK_THRESHOLD}</td>
                <td>{_render_status_pill(count, username in blocked)}</td>
                <td><div class="mini-bar"><span style="width: {width}%"></span></div></td>
            </tr>
        """)

    if not user_rows:
        user_rows.append('<tr><td colspan="4">No user offenses recorded yet.</td></tr>')

    report_rows = []
    for index, report in enumerate(reversed(reports[-12:]), start=1):
        message = escape(report.get("message", ""))
        username = escape(report.get("username") or "Unknown")
        terms = ", ".join(item.get("term", "") for item in report.get("bullying_terms", []))
        terms = escape(terms) if terms else "Not recorded"
        report_rows.append(f"""
            <tr>
                <td>#{index}</td>
                <td>{username}</td>
                <td>{message}</td>
                <td>{terms}</td>
                <td><span class="pill danger">Reported</span></td>
            </tr>
        """)

    if not report_rows:
        report_rows.append('<tr><td colspan="5">No reports have been saved yet.</td></tr>')

    word_bars = []
    max_word = max([count for _, count in top_words], default=1)
    for word, count in top_words:
        width = max(10, int((count / max_word) * 100))
        word_bars.append(f"""
            <div class="bar-row">
                <span>{escape(word)}</span>
                <div><strong style="width: {width}%"></strong></div>
                <em>{count}</em>
            </div>
        """)

    if not word_bars:
        word_bars.append('<p class="empty-state">No report keywords available yet.</p>')

    if performance["available"]:
        matrix = performance["confusion_matrix"]
        performance_html = f"""
            <div class="stats performance-stats">
                {_render_stat_card("Accuracy", _format_metric(performance["accuracy"]), f'{performance["dataset_size"]} labeled samples')}
                {_render_stat_card("Precision", _format_metric(performance["weighted_precision"]), "Weighted average")}
                {_render_stat_card("Recall", _format_metric(performance["weighted_recall"]), "Weighted average")}
                {_render_stat_card("F1 Score", _format_metric(performance["weighted_f1"]), "Weighted average")}
            </div>
            <div class="grid-2 performance-grid">
                <div class="panel">
                    <h4>Confusion Matrix</h4>
                    <table class="matrix">
                        <thead>
                            <tr>
                                <th>Actual / Predicted</th>
                                <th>Safe</th>
                                <th>Bullying</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th>Safe</th>
                                <td>{matrix[0][0]}</td>
                                <td>{matrix[0][1]}</td>
                            </tr>
                            <tr>
                                <th>Bullying</th>
                                <td>{matrix[1][0]}</td>
                                <td>{matrix[1][1]}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="panel">
                    <h4>Class Scores</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Safe</td>
                                <td>{_format_metric(performance["safe"]["precision"])}</td>
                                <td>{_format_metric(performance["safe"]["recall"])}</td>
                                <td>{_format_metric(performance["safe"]["f1-score"])}</td>
                                <td>{int(performance["safe"]["support"])}</td>
                            </tr>
                            <tr>
                                <td>Bullying</td>
                                <td>{_format_metric(performance["bullying"]["precision"])}</td>
                                <td>{_format_metric(performance["bullying"]["recall"])}</td>
                                <td>{_format_metric(performance["bullying"]["f1-score"])}</td>
                                <td>{int(performance["bullying"]["support"])}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        """
    else:
        performance_html = f'<div class="panel"><p class="empty-state">{escape(performance["message"])}</p></div>'

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Cyberbullying Admin Dashboard</title>
        <style>
            :root {{
                --bg: #f5f7fb;
                --panel: #ffffff;
                --text: #1c2430;
                --muted: #637083;
                --line: #dce3ee;
                --primary: #2563eb;
                --teal: #0f9f8f;
                --amber: #d97706;
                --danger: #dc2626;
                --ink: #111827;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: Arial, Helvetica, sans-serif;
            }}
            .shell {{
                display: grid;
                grid-template-columns: 240px 1fr;
                min-height: 100vh;
            }}
            aside {{
                background: var(--ink);
                color: white;
                padding: 24px 18px;
                position: sticky;
                top: 0;
                height: 100vh;
            }}
            aside h1 {{
                font-size: 20px;
                line-height: 1.25;
                margin: 0 0 28px;
            }}
            nav a {{
                color: #dbeafe;
                display: flex;
                gap: 10px;
                padding: 11px 12px;
                margin-bottom: 6px;
                text-decoration: none;
                border-radius: 8px;
                font-size: 14px;
            }}
            nav a:hover, nav a:focus, nav a.active {{ background: rgba(255,255,255,0.14); }}
            main {{
                padding: 28px;
                max-width: 1240px;
                width: 100%;
            }}
            .topbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                margin-bottom: 24px;
            }}
            .topbar h2 {{
                margin: 0 0 6px;
                font-size: 28px;
            }}
            .topbar p {{
                margin: 0;
                color: var(--muted);
            }}
            .button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 40px;
                padding: 0 16px;
                background: var(--primary);
                color: white;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 700;
                white-space: nowrap;
            }}
            .button.secondary {{
                background: #e8eef8;
                color: var(--text);
            }}
            section {{
                display: none;
                margin: 0 0 28px;
                scroll-margin-top: 18px;
            }}
            section.active {{
                display: block;
            }}
            section > h3 {{
                margin: 0 0 14px;
                font-size: 20px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
            }}
            .stat-card, .panel {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                box-shadow: 0 8px 26px rgba(20, 32, 52, 0.06);
            }}
            .stat-card {{ padding: 18px; }}
            .stat-card span, .stat-card small {{
                display: block;
                color: var(--muted);
                font-size: 13px;
            }}
            .stat-card strong {{
                display: block;
                margin: 10px 0 8px;
                font-size: 30px;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
                gap: 16px;
            }}
            .panel {{ padding: 18px; overflow: hidden; }}
            .panel h4 {{
                margin: 0 0 14px;
                font-size: 16px;
            }}
            .check-form {{
                display: grid;
                grid-template-columns: minmax(180px, 0.35fr) minmax(260px, 1fr) auto;
                gap: 12px;
                align-items: end;
            }}
            label {{
                display: grid;
                gap: 7px;
                color: var(--muted);
                font-size: 13px;
                font-weight: 700;
            }}
            input, textarea {{
                width: 100%;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 11px 12px;
                color: var(--text);
                font: inherit;
                background: #fbfdff;
            }}
            textarea {{
                min-height: 88px;
                resize: vertical;
            }}
            .result-card {{
                display: none;
                margin-top: 16px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #fbfdff;
                padding: 16px;
            }}
            .result-card.show {{
                display: block;
            }}
            .result-card.safe {{
                border-color: #99f6e4;
                background: #f0fdfa;
            }}
            .result-card.danger {{
                border-color: #fecaca;
                background: #fff1f2;
            }}
            .result-head {{
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: flex-start;
                margin-bottom: 12px;
            }}
            .result-head strong {{
                display: block;
                font-size: 18px;
                margin-bottom: 5px;
            }}
            .highlighted-message {{
                border: 1px solid var(--line);
                border-radius: 8px;
                background: white;
                padding: 12px;
                line-height: 1.6;
                margin: 12px 0;
                word-break: break-word;
            }}
            mark {{
                background: #fde68a;
                color: #78350f;
                border-radius: 4px;
                padding: 1px 3px;
            }}
            .term-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }}
            .server-response {{
                margin-top: 16px;
                border-top: 1px solid var(--line);
                padding-top: 14px;
            }}
            .server-response h4 {{
                margin: 0 0 12px;
            }}
            .response-table {{
                display: grid;
                grid-template-columns: 70px 1fr;
                border-top: 1px solid var(--line);
                margin-bottom: 12px;
            }}
            .response-table div {{
                padding: 10px 0;
                border-bottom: 1px solid var(--line);
                font-size: 14px;
            }}
            .response-table div:nth-child(odd) {{
                font-weight: 800;
            }}
            pre.response-code {{
                margin: 8px 0 12px;
                padding: 12px;
                border-radius: 6px;
                background: #2f2f2f;
                color: #f8fafc;
                overflow: auto;
                font-size: 13px;
                line-height: 1.45;
                white-space: pre-wrap;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th, td {{
                padding: 12px 10px;
                border-bottom: 1px solid var(--line);
                text-align: left;
                vertical-align: middle;
            }}
            th {{
                color: var(--muted);
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .pill {{
                display: inline-flex;
                align-items: center;
                min-height: 26px;
                padding: 0 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
            }}
            .pill.danger {{ color: #991b1b; background: #fee2e2; }}
            .pill.warning {{ color: #92400e; background: #fef3c7; }}
            .pill.muted {{ color: #374151; background: #e5e7eb; }}
            .mini-bar, .bar-row div {{
                width: 100%;
                height: 9px;
                background: #e8edf5;
                border-radius: 999px;
                overflow: hidden;
            }}
            .mini-bar span, .bar-row strong {{
                display: block;
                height: 100%;
                background: var(--primary);
                border-radius: inherit;
            }}
            .chart-wrap {{
                display: grid;
                grid-template-columns: 220px 1fr;
                align-items: center;
                gap: 18px;
            }}
            .pie {{
                width: 180px;
                aspect-ratio: 1;
                border-radius: 50%;
                background: conic-gradient(
                    var(--danger) 0 {blocked_slice}%,
                    var(--amber) {blocked_slice}% {blocked_slice + warning_slice}%,
                    var(--teal) {blocked_slice + warning_slice}% 100%
                );
                margin: auto;
                position: relative;
            }}
            .pie::after {{
                content: "";
                position: absolute;
                inset: 42px;
                background: white;
                border-radius: 50%;
            }}
            .legend {{
                display: grid;
                gap: 10px;
                color: var(--muted);
                font-size: 14px;
            }}
            .legend span {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .red {{ background: var(--danger); }}
            .amber {{ background: var(--amber); }}
            .teal {{ background: var(--teal); }}
            .bar-row {{
                display: grid;
                grid-template-columns: 90px 1fr 36px;
                gap: 10px;
                align-items: center;
                margin: 12px 0;
                font-size: 14px;
            }}
            .bar-row strong {{ background: var(--teal); }}
            .bar-row em {{
                color: var(--muted);
                font-style: normal;
                text-align: right;
            }}
            .performance-stats {{
                margin-bottom: 16px;
            }}
            .performance-grid {{
                align-items: start;
            }}
            .matrix th, .matrix td {{
                text-align: center;
            }}
            .matrix td {{
                font-size: 24px;
                font-weight: 800;
                color: var(--primary);
            }}
            .settings-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
            }}
            .setting-item {{
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 14px;
                background: #fbfdff;
            }}
            .setting-item span {{
                display: block;
                color: var(--muted);
                font-size: 13px;
                margin-bottom: 8px;
            }}
            .setting-item strong {{
                display: block;
                color: var(--text);
                word-break: break-word;
            }}
            .empty-state {{ color: var(--muted); margin: 0; }}
            @media (max-width: 920px) {{
                .shell {{ grid-template-columns: 1fr; }}
                aside {{
                    position: static;
                    height: auto;
                }}
                nav {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 6px;
                }}
                main {{ padding: 20px; }}
                .stats, .grid-2, .settings-grid, .chart-wrap {{
                    grid-template-columns: 1fr;
                }}
                .check-form {{
                    grid-template-columns: 1fr;
                }}
                .topbar {{ align-items: flex-start; flex-direction: column; }}
            }}
        </style>
    </head>
    <body>
        <div class="shell">
            <aside>
                <h1>Cyberbullying Detection System</h1>
                <nav aria-label="Admin sections">
                    <a class="active" href="#post-comment" data-section="post-comment">Post Comment</a>
                    <a href="#dashboard" data-section="dashboard">Dashboard</a>
                    <a href="#users" data-section="users">User Management</a>
                    <a href="#reports" data-section="reports">Reports</a>
                    <a href="#analytics" data-section="analytics">Analytics</a>
                    <a href="#performance" data-section="performance">Model Performance</a>
                    <a href="#settings" data-section="settings">Settings</a>
                </nav>
            </aside>
            <main>
                <div class="topbar">
                    <div>
                        <h2>Admin Dashboard</h2>
                        <p>Monitor reported messages, user warnings, blocked users, and model activity.</p>
                    </div>
                    <a class="button" href="/export_report">Download PDF Report</a>
                </div>

                <section id="post-comment" class="active">
                    <h3>Post Comment</h3>
                    <div class="panel">
                        <h4>Detect Cyberbullying</h4>
                        <form class="check-form" id="check-form">
                            <label>
                                Username
                                <input id="check-username" name="username" autocomplete="off" required placeholder="Enter username">
                            </label>
                            <label>
                                Comment
                                <textarea id="check-text" name="text" required placeholder="Enter a comment to check"></textarea>
                            </label>
                            <button class="button" type="submit">Check Message</button>
                        </form>
                        <div class="result-card" id="check-result" aria-live="polite"></div>
                    </div>
                </section>

                <section id="dashboard">
                    <h3>Dashboard</h3>
                    <div class="stats">
                        {_render_stat_card("Reported Messages", summary["total_reported_messages"], "Stored abusive comments")}
                        {_render_stat_card("Blocked Users", summary["total_blocked_users"], "Users at 3 warnings")}
                        {_render_stat_card("Offense Events", summary["total_offense_events"], "Total detected violations")}
                        {_render_stat_card("Active Warnings", summary["users_with_warnings"], "Users below block limit")}
                    </div>
                </section>

                <section id="users">
                    <h3>User Management</h3>
                    <div class="panel">
                        <h4>Warning and Block Status</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>User</th>
                                    <th>Warnings</th>
                                    <th>Status</th>
                                    <th>Progress</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(user_rows)}</tbody>
                        </table>
                    </div>
                </section>

                <section id="reports">
                    <h3>Reports</h3>
                    <div class="panel">
                        <h4>Recent Reported Messages</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>User</th>
                                    <th>Message</th>
                                    <th>Detected Terms</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(report_rows)}</tbody>
                        </table>
                    </div>
                </section>

                <section id="analytics">
                    <h3>Analytics</h3>
                    <div class="grid-2">
                        <div class="panel">
                            <h4>Case Distribution</h4>
                            <div class="chart-wrap">
                                <div class="pie" aria-label="Pie chart for reports, warnings, and blocked users"></div>
                                <div class="legend">
                                    <div><span class="red"></span>Blocked users: {blocked_pct}%</div>
                                    <div><span class="amber"></span>Active warnings: {warning_pct}%</div>
                                    <div><span class="teal"></span>Reported messages: {reported_pct}%</div>
                                </div>
                            </div>
                        </div>
                        <div class="panel">
                            <h4>Top Report Keywords</h4>
                            {''.join(word_bars)}
                        </div>
                    </div>
                </section>

                <section id="performance">
                    <h3>Model Performance</h3>
                    {performance_html}
                </section>

                <section id="settings">
                    <h3>Settings</h3>
                    <div class="panel">
                        <div class="settings-grid">
                            <div class="setting-item">
                                <span>Blocking threshold</span>
                                <strong>{BLOCK_THRESHOLD} warnings</strong>
                            </div>
                            <div class="setting-item">
                                <span>Reports file</span>
                                <strong>{escape(REPORT_FILE)}</strong>
                            </div>
                            <div class="setting-item">
                                <span>Model status</span>
                                <strong>{"Loaded" if clf else "Unavailable"}</strong>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </div>
        <script>
            const navLinks = document.querySelectorAll("nav a[data-section]");
            const sections = document.querySelectorAll("main section");
            const form = document.getElementById("check-form");
            const result = document.getElementById("check-result");
            const textInput = document.getElementById("check-text");

            function showSection(sectionId) {{
                sections.forEach((section) => {{
                    section.classList.toggle("active", section.id === sectionId);
                }});
                navLinks.forEach((link) => {{
                    link.classList.toggle("active", link.dataset.section === sectionId);
                }});
                history.replaceState(null, "", "#" + sectionId);
            }}

            navLinks.forEach((link) => {{
                link.addEventListener("click", (event) => {{
                    event.preventDefault();
                    showSection(link.dataset.section);
                }});
            }});

            const initialSection = location.hash.replace("#", "") || "post-comment";
            if (document.getElementById(initialSection)) {{
                showSection(initialSection);
            }}

            function escapeHtml(value) {{
                return value
                    .replaceAll("&", "&amp;")
                    .replaceAll("<", "&lt;")
                    .replaceAll(">", "&gt;")
                    .replaceAll('"', "&quot;")
                    .replaceAll("'", "&#039;");
            }}

            function escapeRegExp(value) {{
                return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
            }}

            function highlightedMessage(message, terms) {{
                let html = escapeHtml(message);
                const sortedTerms = [...terms]
                    .map((item) => item.term)
                    .filter(Boolean)
                    .sort((a, b) => b.length - a.length);

                sortedTerms.forEach((term) => {{
                    const pattern = new RegExp("\\\\b(" + escapeRegExp(escapeHtml(term)) + ")\\\\b", "gi");
                    html = html.replace(pattern, "<mark>$1</mark>");
                }});

                return html;
            }}

            function renderTerms(terms) {{
                if (!terms.length) {{
                    return '<p class="empty-state">No strong bullying terms were identified by the model.</p>';
                }}

                return '<div class="term-list">' + terms.map((item) => (
                    '<span class="pill warning">' + escapeHtml(item.term) + '</span>'
                )).join("") + '</div>';
            }}

            function renderServerResponse(statusCode, body, headers) {{
                const headerText = headers.length
                    ? headers.map(([key, value]) => key + ": " + value).join("\\n")
                    : "No response headers available.";

                return '' +
                    '<div class="server-response">' +
                        '<h4>Server response</h4>' +
                        '<div class="response-table">' +
                            '<div>Code</div>' +
                            '<div>Details</div>' +
                            '<div>' + statusCode + '</div>' +
                            '<div>' +
                                '<strong>Response body</strong>' +
                                '<pre class="response-code">' + escapeHtml(JSON.stringify(body, null, 2)) + '</pre>' +
                                '<strong>Response headers</strong>' +
                                '<pre class="response-code">' + escapeHtml(headerText) + '</pre>' +
                            '</div>' +
                        '</div>' +
                    '</div>';
            }}

            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                const username = document.getElementById("check-username").value.trim();
                const text = textInput.value.trim();

                if (!username || !text) {{
                    return;
                }}

                result.className = "result-card show";
                result.innerHTML = '<p class="empty-state">Checking message with the trained model...</p>';

                try {{
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{"Content-Type": "application/json"}},
                        body: JSON.stringify({{username, text}}),
                    }});
                    const data = await response.json();
                    const responseHeaders = Array.from(response.headers.entries());
                    const isBullying = data.prediction === 1 || Boolean(data.status);
                    const terms = data.bullying_terms || [];
                    const confidence = typeof data.confidence === "number"
                        ? Math.round(data.confidence * 1000) / 10 + "%"
                        : "Blocked user";

                    result.className = "result-card show " + (isBullying ? "danger" : "safe");
                    result.innerHTML =
                        '<div class="result-head">' +
                            '<div>' +
                                '<strong>' + escapeHtml(data.response || data.status || "Result") + '</strong>' +
                                '<span>' + escapeHtml(data.action || data.message || "") + '</span>' +
                            '</div>' +
                            '<span class="pill ' + (isBullying ? "danger" : "muted") + '">' + confidence + '</span>' +
                        '</div>' +
                        '<div class="highlighted-message">' + highlightedMessage(text, terms) + '</div>' +
                        '<h4>Bullying Words Detected</h4>' +
                        renderTerms(terms) +
                        '<p class="empty-state" style="margin-top:12px">Dashboard counts update after refreshing this page.</p>' +
                        renderServerResponse(response.status, data, responseHeaders);
                }} catch (error) {{
                    result.className = "result-card show danger";
                    result.innerHTML = '<p>Could not check the message. Please make sure the API server is running.</p>';
                }}
            }});
        </script>
    </body>
    </html>
    """


# ----------------- Core Prediction Endpoint -----------------
@app.post("/predict")
def predict(payload: TextIn):
    username = payload.username
    text = payload.text

    # Check if user is already blocked
    if is_blocked(username):
        return {
            "username": username,
            "status": "🚫 User already blocked",
            "message": "Your actions violated safety rules. You cannot post further comments.",
            "bullying_terms": [],
        }

    # Clean & predict
    clean = simple_clean_text(text)
    X = vec.transform([clean])
    pred = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][1])

    # If not bullying
    if pred == 0:
        return {
            "username": username,
            "prediction": pred,
            "confidence": prob,
            "response": "✔ Safe message.",
            "action": "Comment posted successfully.",
            "bullying_terms": [],
        }

    bullying_terms = identify_bullying_terms(text)

    # If bullying
    save_report(text, username, prob, bullying_terms)  # Save abusive content

    offense_count = add_offense(username)

    # Warning System
    if offense_count < 3:
        return {
            "username": username,
            "prediction": pred,
            "confidence": prob,
            "response": "⚠ Cyberbullying detected.",
            "action": f"❌ Comment hidden. Warning {offense_count}/3. Repeated abuse will lead to blocking.",
            "offense_count": offense_count,
            "bullying_terms": bullying_terms,
        }
    else:
        save_to_blocklist(username)  # 3rd strike → block user
        return {
            "username": username,
            "prediction": pred,
            "confidence": prob,
            "response": "⚠ Cyberbullying detected repeatedly.",
            "action": "❌ Comment hidden and 🚫 user blocked (3/3 warnings reached).",
            "offense_count": offense_count,
            "bullying_terms": bullying_terms,
        }


# ----------------- Check Block Status -----------------
@app.get("/is_blocked/{username}")
def check_block(username: str):
    return {"username": username, "blocked": is_blocked(username)}


# ----------------- Dashboard -----------------
@app.get("/dashboard")
def dashboard():
    reports = _read_json(REPORT_FILE, [])
    blocked = _read_json(BLOCK_FILE, [])
    offenses = get_offense_counts()

    return {
        "summary": {
            "total_reported_messages": len(reports),
            "total_blocked_users": len(blocked),
            "total_offense_events": sum(offenses.values())
        },
        "blocked_users": blocked,
        "offense_counts": offenses,
    }


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    return HTMLResponse(_render_admin_page())


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_render_admin_page())


@app.get("/model_performance")
def model_performance():
    return get_model_performance()


# ----------------- PDF Report Generator -----------------
def generate_pdf_report(pdf_path="cyber_report.pdf"):
    reports = _read_json(REPORT_FILE, [])
    blocked = _read_json(BLOCK_FILE, [])
    offenses = get_offense_counts()

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    textobject = c.beginText(40, height - 50)

    textobject.textLine("Cyberbullying Detection System - Report")
    textobject.textLine("---------------------------------------")
    textobject.textLine("")
    textobject.textLine(f"Total reported messages: {len(reports)}")
    textobject.textLine(f"Total blocked users: {len(blocked)}")
    textobject.textLine(f"Total offense events: {sum(offenses.values())}")
    textobject.textLine("")

    textobject.textLine("Blocked Users:")
    if blocked:
        for u in blocked:
            textobject.textLine(f" - {u}")
    else:
        textobject.textLine(" (None)")
    textobject.textLine("")

    textobject.textLine("Sample Reported Messages:")
    for r in reports[:10]:
        msg = r.get("message", "")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        textobject.textLine(f" - {msg}")

    c.drawText(textobject)
    c.showPage()
    c.save()


@app.get("/export_report")
def export_report():
    pdf_path = "cyber_report.pdf"
    generate_pdf_report(pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename="cyber_report.pdf")
