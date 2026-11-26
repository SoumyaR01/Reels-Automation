"""ReelForge CLI: Command-line interface for the AI Movie RAG & 9:16 Vertical Reel Studio."""

import argparse
import json
import sys
from pathlib import Path

from .core.config import Settings, get_settings
from .rights.registry import RightsRegistry
from .workers.runner import PipelineRunner
from .rag.db import SceneDatabase
from .rag.retrieval import HybridRetriever
from .planner.agent import ReelPlanningAgent
from .planner.auto_mode import AutoReelBatchGenerator
from .planner.schema import ReelPlan
from .eval.guardrails import DeterministicGuardrails
from .eval.rubric import ReelPlanRubricJudge
from .edit.render import render_reel_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelforge",
        description="ReelForge: Production AI Movie RAG & 9:16 Vertical Reel Studio.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Command: run (Auto Reels execution)
    run_parser = subparsers.add_parser("run", help="Run full pipeline to extract and compile reels")
    run_parser.add_argument("input", type=Path, help="Path to input movie file")
    run_parser.add_argument("--output-dir", type=Path, default=Path("reels"), help="Output directory for reels")
    run_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")
    run_parser.add_argument("--categories", nargs="+", default=["sad", "cinematic"], help="Categories to export")
    run_parser.add_argument("--allow-unverified-rights", action="store_true", default=True, help="Bypass publishing rights gate")

    # Command: rag (Question answering and scene discovery)
    rag_parser = subparsers.add_parser("rag", help="Query scene database using semantic search and Question Library")
    rag_parser.add_argument("query", type=str, help="Natural language query or question from library")
    rag_parser.add_argument("--movie-id", default="heart_of_the_beast_2026_1080p_am_83a83fa6", help="Target movie ID")
    rag_parser.add_argument("--top-k", type=int, default=5, help="Number of matching scenes to return")
    rag_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")

    # Command: plan (Reel Planning Agent)
    plan_parser = subparsers.add_parser("plan", help="Generate a structured Section 7 ReelPlan from user prompt")
    plan_parser.add_argument("prompt", type=str, help="User prompt (e.g. 'Create a 30-second motivational reel')")
    plan_parser.add_argument("--movie-id", default="heart_of_the_beast_2026_1080p_am_83a83fa6", help="Target movie ID")
    plan_parser.add_argument("--duration", type=float, default=30.0, help="Target reel duration in seconds")
    plan_parser.add_argument("--with-shayari", action="store_true", default=False, help="Include Hindi shayari overlay")
    plan_parser.add_argument("--output", type=Path, default=None, help="Save plan JSON to file path")
    plan_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")

    # Command: auto (Auto-Reel Batch Mode)
    auto_parser = subparsers.add_parser("auto", help="Auto-generate multiple 30s reel plans across all 6 categories")
    auto_parser.add_argument("--movie-id", default="heart_of_the_beast_2026_1080p_am_83a83fa6", help="Target movie ID")
    auto_parser.add_argument("--reels-per-cat", type=int, default=2, help="Number of reel plans per category")
    auto_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")

    # Command: render (Render an approved ReelPlan to MP4)
    render_parser = subparsers.add_parser("render", help="Render an approved ReelPlan JSON into a 9:16 vertical reel")
    render_parser.add_argument("plan", type=Path, help="Path to ReelPlan JSON file")
    render_parser.add_argument("--movie-file", type=Path, default=Path("Heart.of.the.Beast.2026.1080p.AMZN.WEB-DL.DUAL.DDP5.1.H.264-Vegamovies.tw.mkv"), help="Source movie video file")
    render_parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    render_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")

    # Command: eval (Deterministic checks and rubric evaluation)
    eval_parser = subparsers.add_parser("eval", help="Run Section 12A deterministic checks and rubric evaluation on a ReelPlan")
    eval_parser.add_argument("plan", type=Path, help="Path to ReelPlan JSON file")
    eval_parser.add_argument("--movie-id", default="heart_of_the_beast_2026_1080p_am_83a83fa6", help="Target movie ID")
    eval_parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory")

    # Command: rights
    rights_parser = subparsers.add_parser("rights", help="View or update rights registry clearance")
    rights_parser.add_argument("--source-id", required=True, help="Source movie ID or filename stem")
    rights_parser.add_argument("--status", choices=["licensed", "public_domain", "own", "fair_dealing_commentary", "unknown"], help="Set clearance status")
    rights_parser.add_argument("--holder", default=None, help="Rights holder / studio / distributor")
    rights_parser.add_argument("--notes", default=None, help="Licensing notes or contract references")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI studio server and review dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand or args.subcommand == "run":
        if not hasattr(args, "input") or not args.input:
            parser.print_help()
            sys.exit(1)

        settings = get_settings()
        settings.output_dir = getattr(args, "output_dir", settings.output_dir)
        settings.work_dir = getattr(args, "work_dir", settings.work_dir)

        runner = PipelineRunner(args.input, settings=settings)
        runner.run_all(
            categories=getattr(args, "categories", ["sad", "cinematic"]),
            allow_unverified_rights=getattr(args, "allow_unverified_rights", True),
        )

    elif args.subcommand == "rag":
        db_file = args.work_dir / args.movie_id / "scene_db.json"
        if not db_file.is_file():
            print(f"Error: Scene database not found at {db_file}. Please index movie first.", file=sys.stderr)
            sys.exit(1)
        db = SceneDatabase(db_file)
        retriever = HybridRetriever(db)
        res = retriever.answer_question(args.query)
        print(f"\n🎬 Question: {res['question']}")
        print(f"📊 Summary: {res['answer_summary']}\n")
        print("Top Scene Matches:")
        for idx, m in enumerate(res["top_scene_matches"][:args.top_k], 1):
            print(f"  #{idx:02d} Scene {m['scene_id']} [{m['start_time_seconds']:.1f}s - {m['end_time_seconds']:.1f}s] ({m['duration_seconds']:.1f}s)")
            print(f"       Category: {m['category']} (score: {m['category_score']:.2f}) | Relevance: {m['relevance_score']:.2f}")
            print(f"       Reason: {m['reason']}")
            if m.get("dialogue_excerpt"):
                print(f"       Dialogue: {m['dialogue_excerpt']}")
            print()

    elif args.subcommand == "plan":
        db_file = args.work_dir / args.movie_id / "scene_db.json"
        if not db_file.is_file():
            print(f"Error: Scene database not found at {db_file}.", file=sys.stderr)
            sys.exit(1)
        db = SceneDatabase(db_file)
        agent = ReelPlanningAgent(db)
        plan = agent.plan_reel(
            prompt=args.prompt,
            target_duration=args.duration,
            include_shayari=args.with_shayari,
        )
        print(f"\n📋 Generated ReelPlan: {plan.title} (ID: {plan.plan_id})")
        print(f"⏱️ Duration: {plan.request.target_duration_seconds:.1f}s | Aspect Ratio: {plan.request.aspect_ratio}")
        print(f"🎵 Music: {plan.music_direction}")
        if plan.shayari:
            print(f"\n🪶 Hindi Shayari ({plan.shayari.reel_start_seconds}s-{plan.shayari.reel_end_seconds}s) [AI-Generated]:")
            print(f"   {plan.shayari.text.replace(chr(10), ' / ')}")
        print("\nTimeline Beats:")
        for idx, c in enumerate(plan.selected_clips, 1):
            print(f"  Beat #{idx}: [{c.reel_start_seconds:.1f}s - {c.reel_end_seconds:.1f}s] -> Source {c.scene_id} [{c.source_start_seconds:.1f}s - {c.source_end_seconds:.1f}s] ({c.reason})")

        out_path = args.output or Path("reels") / f"{plan.plan_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))
        print(f"\n✅ Saved Plan JSON: {out_path.resolve()}")

    elif args.subcommand == "auto":
        db_file = args.work_dir / args.movie_id / "scene_db.json"
        if not db_file.is_file():
            print(f"Error: Scene database not found at {db_file}.", file=sys.stderr)
            sys.exit(1)
        db = SceneDatabase(db_file)
        generator = AutoReelBatchGenerator(db, output_queue_dir=Path("reels/review_queue"))
        print(f"Running Section 2A Auto-Reel Batch Mode across 6 categories...")
        results = generator.generate_all_categories(reels_per_category=args.reels_per_cat)
        total_generated = sum(len(plans) for plans in results.values())
        print(f"\n🎉 Successfully generated {total_generated} ReelPlans in 'reels/review_queue/':")
        for cat, plans in results.items():
            print(f"  • {cat.upper()}: {len(plans)} plans")
            for p in plans:
                print(f"      - {p.plan_id}: {p.title} ({p.request.target_duration_seconds:.1f}s)")

    elif args.subcommand == "render":
        if not args.plan.is_file():
            print(f"Error: Plan file not found: {args.plan}", file=sys.stderr)
            sys.exit(1)
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = ReelPlan(**json.load(f))
        out_path = args.output or Path("reels") / f"{plan.plan_id}.mp4"
        print(f"Rendering ReelPlan '{plan.title}' to {out_path} ...")
        rendered = render_reel_plan(
            input_path=args.movie_file,
            plan=plan,
            output_path=out_path,
            work_dir=args.work_dir / "render_tmp",
        )
        print(f"✅ Rendered successfully: {rendered.resolve()}")

    elif args.subcommand == "eval":
        if not args.plan.is_file():
            print(f"Error: Plan file not found: {args.plan}", file=sys.stderr)
            sys.exit(1)
        db_file = args.work_dir / args.movie_id / "scene_db.json"
        db = SceneDatabase(db_file) if db_file.is_file() else SceneDatabase()
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = ReelPlan(**json.load(f))
        eval_res = ReelPlanRubricJudge.evaluate(plan, db)
        print(f"\n🔍 Section 12A Evaluation Report for '{plan.title}':")
        print(f"Overall Result: {'✅ PASSED' if eval_res['overall_pass'] else '❌ FAILED'}")
        print(f"Composite Score: {eval_res['composite_score']:.2f} / 5.00")
        print("\nRubric Dimensions (1-5):")
        for dim, score in eval_res["rubric_scores"].items():
            print(f"  • {dim.replace('_', ' ').title():24s}: {score:.1f} / 5.0")
        print("\nDeterministic Checks:")
        for check, passed in eval_res["deterministic_checks"].items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check}")
        if eval_res["violations"]:
            print("\nViolations:")
            for v in eval_res["violations"]:
                print(f"  ⚠️ {v}")

    elif args.subcommand == "rights":
        settings = get_settings()
        reg = RightsRegistry(settings.work_dir / "rights_registry.json")
        if args.status:
            info = reg.set_rights(
                source_id=args.source_id,
                status=args.status,
                rights_holder=args.holder,
                notes=args.notes,
            )
            print(f"Updated rights for '{args.source_id}': status='{info.source_status}', holder='{info.rights_holder}'")
        else:
            info = reg.get_rights(args.source_id)
            print(f"Rights status for '{args.source_id}': {info.model_dump_json(indent=2)}")

    elif args.subcommand == "serve":
        import uvicorn
        if sys.platform == "win32":
            import asyncio
            try:
                # Use SelectorEventLoop on Windows to avoid WinError 10054 ProactorBasePipeTransport crashes
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        from .api.app import create_app
        print(f"Starting ReelForge Studio on http://{args.host}:{args.port}/review/ ...")
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", loop="asyncio")


if __name__ == "__main__":
    main()
