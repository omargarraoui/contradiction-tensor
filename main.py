import sys
import uuid

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich import box

from tensor import ContradictionTensor, Belief
from viz import heatmap


console = Console()


DEMO_BELIEFS = [
    ("fed_hawk", "The Fed will keep rates high for longer", "macro"),
    ("risk_on", "Now is a good time to be long risk assets", "macro"),
    ("dlr_str", "The dollar will strengthen significantly", "macro"),
    ("em_bull", "Emerging markets will outperform this year", "macro"),
    ("sft_lnd", "The economy is heading for a soft landing", "macro"),
]


def _print_beliefs(ct: ContradictionTensor) -> None:
    if not ct.beliefs:
        console.print("[dim]no beliefs added yet[/dim]")
        return
    t = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    t.add_column("id", style="dim", no_wrap=True)
    t.add_column("belief")
    t.add_column("domain", style="dim")
    for b in ct.beliefs:
        t.add_row(b.id, b.text, b.domain)
    console.print(t)


def _print_summary(ct: ContradictionTensor) -> None:
    s = ct.summary()
    console.print(
        f"  beliefs [bold]{s['beliefs']}[/bold]  "
        f"mean [bold]{s['mean_tension']:.3f}[/bold]  "
        f"max [bold]{s['max_tension']:.3f}[/bold]  "
        f"p90 [bold]{s['p90_tension']:.3f}[/bold]"
    )


HELP = """
  [bold]add[/bold]       add a belief
  [bold]list[/bold]      show all beliefs
  [bold]max[/bold]       highest contradiction pair
  [bold]remove[/bold]    min beliefs to remove for consistency
  [bold]clusters[/bold]  show contradiction clusters
  [bold]heatmap[/bold]   save heatmap.png
  [bold]summary[/bold]   tension stats
  [bold]demo[/bold]      load demo beliefs
  [bold]quit[/bold]      exit
"""


def main() -> None:
    contexts = ["default", "crisis", "bull_market"]
    ct = ContradictionTensor(contexts=contexts)

    console.print(Panel("[bold]Contradiction Tensor[/bold]", expand=False))
    console.print(f"  contexts: {', '.join(contexts)}")
    console.print("  type [dim]help[/dim] for commands\n")

    while True:
        try:
            cmd = Prompt.ask("[cyan]>[/cyan]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd == "help":
            console.print(HELP)

        elif cmd == "add":
            text = Prompt.ask("  belief")
            domain = Prompt.ask("  domain", default="general")
            bid = text[:6].lower().replace(" ", "_") + "_" + uuid.uuid4().hex[:4]
            crisis_w = float(Prompt.ask("  crisis weight [0-2]", default="1.0"))
            bull_w = float(Prompt.ask("  bull market weight [0-2]", default="1.0"))

            with console.status("  scoring against existing beliefs..."):
                ct.add(
                    Belief(id=bid, text=text, domain=domain),
                    context_weights=[1.0, crisis_w, bull_w],
                )
            console.print(f"  [green]added[/green] [dim]{bid}[/dim]")

        elif cmd == "demo":
            with console.status("  loading demo beliefs..."):
                for bid, text, domain in DEMO_BELIEFS:
                    ct.add(Belief(id=bid, text=text, domain=domain))
            console.print(f"  loaded {len(DEMO_BELIEFS)} beliefs")

        elif cmd == "list":
            _print_beliefs(ct)

        elif cmd == "summary":
            if len(ct.beliefs) < 2:
                console.print("  [dim]need at least 2 beliefs[/dim]")
            else:
                _print_summary(ct)

        elif cmd == "max":
            result = ct.max_contradiction()
            if not result:
                console.print("  [dim]need at least 2 beliefs[/dim]")
                continue
            a, b, score = result
            bar = "█" * int(score * 20)
            console.print(f"\n  [red]{score:.3f}[/red]  {bar}")
            console.print(f"  → {a.text}  [dim]({a.id})[/dim]")
            console.print(f"  → {b.text}  [dim]({b.id})[/dim]\n")

            for ctx in ct.contexts[1:]:
                r = ct.max_contradiction(context=ctx)
                if r:
                    ca, cb, cs = r
                    console.print(
                        f"  [dim]{ctx}[/dim]  [red]{cs:.3f}[/red]  "
                        f"{ca.id} ↔ {cb.id}"
                    )
            console.print()

        elif cmd == "remove":
            if len(ct.beliefs) < 2:
                console.print("  [dim]need at least 2 beliefs[/dim]")
                continue
            try:
                threshold = float(Prompt.ask("  threshold", default="0.7"))
            except ValueError:
                console.print("  [red]threshold deve essere un numero (es. 0.7)[/red]")
                continue
            removed = ct.min_removal_set(threshold)
            if not removed:
                console.print(f"  [green]no conflicts above {threshold}[/green]")
            else:
                console.print(f"\n  remove to reach consistency (threshold {threshold}):")
                for b in removed:
                    console.print(f"  [red]−[/red] {b.text}  [dim]({b.id})[/dim]")
                console.print()

        elif cmd == "clusters":
            if len(ct.beliefs) < 2:
                console.print("  [dim]need at least 2 beliefs[/dim]")
                continue
            try:
                threshold = float(Prompt.ask("  threshold", default="0.6"))
            except ValueError:
                console.print("  [red]threshold deve essere un numero (es. 0.6)[/red]")
                continue
            clusters = ct.tension_clusters(threshold)
            if not clusters:
                console.print(f"  [green]no clusters above {threshold}[/green]")
            else:
                for i, cluster in enumerate(clusters, 1):
                    console.print(f"  cluster {i}: {', '.join(cluster)}")

        elif cmd == "heatmap":
            if len(ct.beliefs) < 2:
                console.print("  [dim]need at least 2 beliefs[/dim]")
                continue
            path = Prompt.ask("  output path", default="heatmap.png")
            with console.status("  rendering..."):
                heatmap(ct, output_path=path)
            console.print(f"  [green]saved →[/green] {path}")

        elif cmd == "quit":
            break

        else:
            console.print(f"  [dim]unknown command '{cmd}' — type help[/dim]")


if __name__ == "__main__":
    main()