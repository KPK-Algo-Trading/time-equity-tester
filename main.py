import argparse
import sys
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# Trading sessions defined in UTC hours
SESSIONS = {
    "Sydney":    (21, 6),   # 21:00 – 06:00 UTC (crosses midnight)
    "Tokyo":     (0,  9),   # 00:00 – 09:00 UTC
    "London":    (7,  16),  # 07:00 – 16:00 UTC
    "New York":  (12, 21),  # 12:00 – 21:00 UTC
}

SESSION_COLORS = {
    "Sydney":   "#4fc3f7",
    "Tokyo":    "#ce93d8",
    "London":   "#a5d6a7",
    "New York": "#ffcc80",
    "Overlap":  "#ef9a9a",
    "Off-hours":"#546e7a",
}

DARK    = "#0a0e17"
PANEL   = "#111827"
GRID    = "#1f2937"
ACC1    = "#38bdf8"
ACC2    = "#4ade80"
ACC3    = "#f87171"
ACC4    = "#fbbf24"
ACC5    = "#a78bfa"
TEXT    = "#f1f5f9"
MUTED   = "#64748b"

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─────────────────────────────────────────────
#  INPUT HELPERS
# ─────────────────────────────────────────────

def ask(prompt, cast=str, validator=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = cast(raw)
            if validator and not validator(value):
                raise ValueError
            return value
        except (ValueError, TypeError):
            print("  x  Invalid input, please try again.")


def ask_timezone():
    """Prompt user for a valid pytz timezone string."""
    print("\n  Common examples: UTC, US/Eastern, Europe/London, Asia/Tokyo,")
    print("                   America/New_York, Europe/Berlin, Asia/Singapore\n")
    while True:
        tz_str = input("Timezone of trades: ").strip()
        try:
            tz = pytz.timezone(tz_str)
            return tz, tz_str
        except pytz.UnknownTimeZoneError:
            print(f"  x  Unknown timezone '{tz_str}'. Please try again.")


# ─────────────────────────────────────────────
#  DATA LOADING & ENRICHMENT
# ─────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["result"] = df["result"].str.strip().str.upper()
    df["exit_datetime"] = pd.to_datetime(df["exit_datetime"].str.strip())
    return df


def assign_session(utc_hour):
    """Return a list of session names active at a given UTC hour."""
    active = []
    for name, (start, end) in SESSIONS.items():
        if start > end:  # crosses midnight (Sydney)
            if utc_hour >= start or utc_hour < end:
                active.append(name)
        else:
            if start <= utc_hour < end:
                active.append(name)
    if len(active) == 0:
        return "Off-hours"
    if len(active) > 1:
        return "Overlap"
    return active[0]


def enrich(df, local_tz, tz_str):
    """Add local time columns and session labels."""
    # Localise naive timestamps to the user's timezone, then convert to UTC
    df["local_dt"] = df["exit_datetime"].dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
    df["utc_dt"]   = df["local_dt"].dt.tz_convert("UTC")

    df["local_hour"] = df["local_dt"].dt.hour
    df["local_day"]  = df["local_dt"].dt.day_name()
    df["utc_hour"]   = df["utc_dt"].dt.hour

    df["session"]    = df["utc_hour"].apply(assign_session)

    # Numeric outcome for EV calculation
    #   TP = +1, SL = -1, OPEN = 0 (neutral / incomplete)
    df["outcome"] = df["result"].map({"TP": 1, "SL": -1, "OPEN": 0})
    df["is_tp"]   = (df["result"] == "TP").astype(int)
    df["is_sl"]   = (df["result"] == "SL").astype(int)
    df["is_open"] = (df["result"] == "OPEN").astype(int)

    return df.dropna(subset=["local_dt"])


# ─────────────────────────────────────────────
#  METRIC AGGREGATION
# ─────────────────────────────────────────────

def agg_metrics(df, group_col, all_values=None):
    """
    For each unique value of group_col compute:
      trades, win_rate, ev (expected value = mean outcome), tp_count, sl_count, open_count
    Returns a DataFrame indexed by group_col values.
    """
    g = df.groupby(group_col)
    trades   = g["outcome"].count()
    ev       = g["outcome"].mean()
    win_rate = g["is_tp"].sum() / trades * 100
    tp_count = g["is_tp"].sum()
    sl_count = g["is_sl"].sum()
    op_count = g["is_open"].sum()

    out = pd.DataFrame({
        "trades":   trades,
        "win_rate": win_rate,
        "ev":       ev,
        "tp":       tp_count,
        "sl":       sl_count,
        "open":     op_count,
    })

    if all_values is not None:
        out = out.reindex(all_values, fill_value=0)

    return out


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, axis="y")


def bar_triple(ax, x_labels, metrics, title, x_label="", rotate=0):
    """
    Three-metric grouped bar: trades (count), win rate (%), EV.
    Uses a twin axis for win rate and EV overlaid as lines.
    """
    x    = np.arange(len(x_labels))
    width = 0.6

    bars = ax.bar(x, metrics["trades"], width=width,
                  color=ACC1, alpha=0.75, edgecolor=DARK, linewidth=0.4,
                  label="Trades")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center",
                       fontsize=7.5, color=MUTED)
    ax.set_ylabel("Trades", color=MUTED, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=10)

    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL)
    ax2.plot(x, metrics["win_rate"], color=ACC2, linewidth=2,
             marker="o", markersize=4, label="Win rate %", zorder=5)
    ax2.plot(x, metrics["ev"] * 100, color=ACC4, linewidth=1.5,
             marker="s", markersize=3.5, linestyle="--", label="EV x100", zorder=5)
    ax2.axhline(0,  color=MUTED, linewidth=0.6, linestyle=":")
    ax2.axhline(50, color=ACC2,  linewidth=0.5, linestyle=":", alpha=0.4)
    ax2.set_ylabel("Win rate / EV x100 (%)", color=MUTED, fontsize=7)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)

    if x_label:
        ax.set_xlabel(x_label, color=MUTED, fontsize=8)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7,
               facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, loc="upper right")


def stacked_result_bar(ax, x_labels, metrics, title, rotate=0):
    """Stacked bar: TP / SL / OPEN counts per category."""
    x     = np.arange(len(x_labels))
    width = 0.55

    ax.bar(x, metrics["tp"],   width=width, color=ACC2, edgecolor=DARK,
           linewidth=0.4, label="TP")
    ax.bar(x, metrics["sl"],   width=width, bottom=metrics["tp"],
           color=ACC3, edgecolor=DARK, linewidth=0.4, label="SL")
    ax.bar(x, metrics["open"], width=width,
           bottom=metrics["tp"] + metrics["sl"],
           color=MUTED, edgecolor=DARK, linewidth=0.4, label="OPEN", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center",
                       fontsize=7.5, color=MUTED)
    ax.set_ylabel("Count", color=MUTED, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=10)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)


def ev_heatmap(ax, fig, df, title):
    """
    2-D heatmap: rows = day of week, cols = hour of day.
    Cell value = mean EV. Only cells with at least one trade are filled.
    """
    pivot = df.pivot_table(
        index="local_day", columns="local_hour",
        values="outcome", aggfunc="mean"
    ).reindex(DAYS_ORDER)

    hours = np.arange(24)
    pivot = pivot.reindex(columns=hours)

    data = pivot.values.astype(float)

    vmax = max(abs(np.nanmax(data)), abs(np.nanmin(data)), 0.01)
    im   = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(colors=MUTED, labelsize=7)

    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=6, color=MUTED)
    ax.set_yticks(np.arange(len(DAYS_ORDER)))
    ax.set_yticklabels(DAYS_ORDER, fontsize=7.5, color=MUTED)
    ax.set_xlabel("Local hour", color=MUTED, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=10)

    # Annotate cells that have data
    count_pivot = df.pivot_table(
        index="local_day", columns="local_hour",
        values="outcome", aggfunc="count"
    ).reindex(DAYS_ORDER).reindex(columns=hours)

    for r in range(len(DAYS_ORDER)):
        for c in range(24):
            val = data[r, c]
            cnt = count_pivot.iloc[r, c] if not count_pivot.empty else np.nan
            if not np.isnan(val):
                ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(val) > vmax * 0.5 else DARK,
                        fontweight="bold")


def session_pie(ax, df):
    """Pie chart of trade distribution across sessions."""
    counts = df["session"].value_counts()
    colors = [SESSION_COLORS.get(s, MUTED) for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor=DARK, linewidth=0.8),
        textprops=dict(color=TEXT, fontsize=8),
    )
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_color(DARK)
    ax.set_title("Trade Distribution by Session", color=TEXT, fontsize=10)


def winrate_ev_session(ax, metrics_sess):
    """Horizontal bar chart: win rate and EV per session."""
    sessions = metrics_sess.index.tolist()
    y        = np.arange(len(sessions))
    height   = 0.35

    ax.barh(y + height / 2, metrics_sess["win_rate"], height=height,
            color=ACC2, alpha=0.8, edgecolor=DARK, linewidth=0.4, label="Win rate %")
    ax.barh(y - height / 2, metrics_sess["ev"] * 100, height=height,
            color=ACC4, alpha=0.8, edgecolor=DARK, linewidth=0.4, label="EV x100")

    ax.axvline(0,  color=MUTED, linewidth=0.8, linestyle=":")
    ax.axvline(50, color=ACC2,  linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(sessions, fontsize=8, color=MUTED)
    ax.set_xlabel("Value", color=MUTED, fontsize=8)
    ax.set_title("Win Rate & EV per Session", color=TEXT, fontsize=10)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    ax.grid(True, color=GRID, linewidth=0.5, axis="x")


def summary_panel(ax, df, tz_str):
    ax.axis("off")
    ax.set_facecolor(PANEL)

    total   = len(df)
    tp_n    = (df["result"] == "TP").sum()
    sl_n    = (df["result"] == "SL").sum()
    op_n    = (df["result"] == "OPEN").sum()
    wr      = tp_n / total * 100 if total else 0
    ev      = df["outcome"].mean()
    best_h  = df.groupby("local_hour")["outcome"].mean().idxmax() if total else "-"
    best_d  = df.groupby("local_day")["outcome"].mean().idxmax()  if total else "-"
    best_s  = df.groupby("session")["outcome"].mean().idxmax()    if total else "-"

    date_range = (f"{df['local_dt'].min().strftime('%Y-%m-%d')} to "
                  f"{df['local_dt'].max().strftime('%Y-%m-%d')}")

    lines = [
        ("OVERVIEW",      None,                    ACC1),
        ("Timezone",       tz_str,                  None),
        ("Date range",     date_range,              None),
        ("Total trades",   f"{total:,}",            None),
        ("TP",             f"{tp_n:,} ({tp_n/total*100:.1f}%)" if total else "0", None),
        ("SL",             f"{sl_n:,} ({sl_n/total*100:.1f}%)" if total else "0", None),
        ("OPEN",           f"{op_n:,} ({op_n/total*100:.1f}%)" if total else "0", None),
        ("Win rate",       f"{wr:.1f}%",            None),
        ("EV (raw)",       f"{ev:+.3f}",            None),
        ("",               None,                    None),
        ("BEST PERIODS",  None,                    ACC2),
        ("Best hour",      f"{best_h:02d}:00" if isinstance(best_h, int) else str(best_h), None),
        ("Best day",       str(best_d),             None),
        ("Best session",   str(best_s),             None),
    ]

    y    = 0.97
    step = 0.067

    for label, value, color in lines:
        if value is None:
            if label:
                ax.text(0.02, y, label, transform=ax.transAxes,
                        color=color or TEXT, fontsize=9, fontweight="bold")
            y -= step
            continue
        ax.text(0.02, y, label, transform=ax.transAxes, color=MUTED, fontsize=8)
        ax.text(0.98, y, value, transform=ax.transAxes, color=TEXT,
                fontsize=8, ha="right", fontweight="bold")
        y -= step

    ax.set_title("Summary", color=TEXT, fontsize=10)


# ─────────────────────────────────────────────
#  MAIN PLOT ORCHESTRATOR
# ─────────────────────────────────────────────

def plot_all(df, tz_str):
    all_hours   = list(range(24))
    all_days    = DAYS_ORDER
    all_sessions = list(SESSIONS.keys()) + ["Overlap", "Off-hours"]

    m_hour    = agg_metrics(df, "local_hour",  all_hours)
    m_day     = agg_metrics(df, "local_day",   all_days)
    m_session = agg_metrics(df, "session",     all_sessions)
    # Drop sessions with zero trades for cleaner plots
    m_session = m_session[m_session["trades"] > 0]

    fig = plt.figure(figsize=(24, 18), facecolor=DARK)
    fig.suptitle(
        f"Trade Time Analysis  —  timezone: {tz_str}",
        color=TEXT, fontsize=16, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(
        4, 4,
        figure=fig,
        hspace=0.55, wspace=0.40,
        left=0.05, right=0.97, top=0.96, bottom=0.04
    )

    # Row 0: hourly triple-bar | hourly stacked result
    ax_h_triple  = fig.add_subplot(gs[0, :2])
    ax_h_stack   = fig.add_subplot(gs[0, 2:])

    # Row 1: daily triple-bar | daily stacked result
    ax_d_triple  = fig.add_subplot(gs[1, :2])
    ax_d_stack   = fig.add_subplot(gs[1, 2:])

    # Row 2: EV heatmap (full width)
    ax_heatmap   = fig.add_subplot(gs[2, :])

    # Row 3: session pie | session winrate/ev | summary
    ax_pie       = fig.add_subplot(gs[3, 0])
    ax_sess_bar  = fig.add_subplot(gs[3, 1:3])
    ax_summary   = fig.add_subplot(gs[3, 3])

    all_axes = [ax_h_triple, ax_h_stack, ax_d_triple, ax_d_stack,
                ax_heatmap, ax_sess_bar, ax_summary]
    for ax in all_axes:
        style_ax(ax)
    ax_pie.set_facecolor(PANEL)
    ax_pie.tick_params(colors=MUTED)
    ax_pie.title.set_color(TEXT)

    # ── Hourly ──────────────────────────────────────────────────────────
    bar_triple(ax_h_triple,
               [f"{h:02d}" for h in all_hours],
               m_hour,
               "Trades / Win Rate / EV  by Hour of Day",
               x_label="Local hour")

    stacked_result_bar(ax_h_stack,
                       [f"{h:02d}" for h in all_hours],
                       m_hour,
                       "TP / SL / OPEN  by Hour of Day")

    # ── Daily ───────────────────────────────────────────────────────────
    day_labels = [d[:3] for d in all_days]   # Mon, Tue …

    bar_triple(ax_d_triple,
               day_labels,
               m_day,
               "Trades / Win Rate / EV  by Day of Week")

    stacked_result_bar(ax_d_stack,
                       day_labels,
                       m_day,
                       "TP / SL / OPEN  by Day of Week")

    # ── EV Heatmap ──────────────────────────────────────────────────────
    ev_heatmap(ax_heatmap, fig, df, "EV Heatmap  —  Day of Week  x  Local Hour")

    # ── Sessions ────────────────────────────────────────────────────────
    session_pie(ax_pie, df)
    winrate_ev_session(ax_sess_bar, m_session)

    # ── Summary ─────────────────────────────────────────────────────────
    summary_panel(ax_summary, df, tz_str)

    plt.savefig("trade_time_analysis.png", dpi=150,
                bbox_inches="tight", facecolor=DARK)
    print("\n  Saved --> trade_time_analysis.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade Time Tester")
    parser.add_argument("--file", required=True, help="Path to trades datetime CSV")
    args = parser.parse_args()

    print("=" * 52)
    print("         TRADE TIME ANALYSIS TESTER")
    print("=" * 52)

    df_raw = load_csv(args.file)
    print(f"\n  Loaded {len(df_raw)} trades from '{args.file}'")

    local_tz, tz_str = ask_timezone()
    df = enrich(df_raw, local_tz, tz_str)

    skipped = len(df_raw) - len(df)
    if skipped:
        print(f"  Warning: {skipped} rows dropped (ambiguous / missing timestamps)")

    print(f"\n  Analysing {len(df)} trades across "
          f"{df['local_day'].nunique()} days of week, "
          f"{df['local_hour'].nunique()} distinct hours, "
          f"{df['session'].nunique()} sessions ...\n")

    # Console summary
    total = len(df)
    tp_n  = (df["result"] == "TP").sum()
    sl_n  = (df["result"] == "SL").sum()
    ev    = df["outcome"].mean()
    print(f"{'─'*52}")
    print(f"  Total trades  : {total:,}")
    print(f"  Win rate      : {tp_n/total*100:.1f}%  ({tp_n} TP / {sl_n} SL)")
    print(f"  EV (raw)      : {ev:+.4f}")
    print(f"{'─'*52}\n")

    print("  Best periods by EV:")
    for label, col in [("hour", "local_hour"), ("day", "local_day"), ("session", "session")]:
        grp = df.groupby(col)["outcome"].mean()
        best_val = grp.idxmax()
        print(f"    Best {label:<9}: {best_val}  (EV {grp[best_val]:+.3f})")
    print()

    plot_all(df, tz_str)


if __name__ == "__main__":
    main()