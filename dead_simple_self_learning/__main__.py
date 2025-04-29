"""
Command-line interface for the dead_simple_self_learning package.

This allows direct interaction with the package functionality from the command line.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

from .embedder import Embedder
from .memory import Memory
from .learner import SelfLearner


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Dead Simple Self-Learning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Embed text command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for text")
    embed_parser.add_argument("text", help="Text to embed")
    embed_parser.add_argument("--model", choices=["openai", "miniLM", "bge-small"], 
                              default="miniLM", help="Embedding model to use")
    
    # Enhance prompt command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance a prompt with relevant feedback")
    enhance_parser.add_argument("task", help="Task description")
    enhance_parser.add_argument("base_prompt", help="Base prompt to enhance")
    enhance_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    enhance_parser.add_argument("--model", choices=["openai", "miniLM", "bge-small"], 
                                default="miniLM", help="Embedding model to use")
    enhance_parser.add_argument("--threshold", type=float, default=0.85, 
                               help="Similarity threshold (0-1)")
    
    # Save feedback command
    save_parser = subparsers.add_parser("save", help="Save feedback for a task")
    save_parser.add_argument("task", help="Task description")
    save_parser.add_argument("feedback", help="Feedback to save")
    save_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    save_parser.add_argument("--model", choices=["openai", "miniLM", "bge-small"], 
                             default="miniLM", help="Embedding model to use")
    
    # View memory command
    view_parser = subparsers.add_parser("view", help="View all feedback in memory")
    view_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    view_parser.add_argument("--format", choices=["text", "json"], default="text",
                             help="Output format")
    
    # Reset memory command
    reset_parser = subparsers.add_parser("reset", help="Reset all feedback memory")
    reset_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    reset_parser.add_argument("--confirm", action="store_true", 
                              help="Confirm reset without prompt")
    
    # Export memory command
    export_parser = subparsers.add_parser("export", help="Export memory to a file")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    
    # Import memory command
    import_parser = subparsers.add_parser("import", help="Import memory from a file")
    import_parser.add_argument("input", help="Input file path")
    import_parser.add_argument("--memory", default="memory.json", help="Path to memory file")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "embed":
        handle_embed(args)
    elif args.command == "enhance":
        handle_enhance(args)
    elif args.command == "save":
        handle_save(args)
    elif args.command == "view":
        handle_view(args)
    elif args.command == "reset":
        handle_reset(args)
    elif args.command == "export":
        handle_export(args)
    elif args.command == "import":
        handle_import(args)
    elif args.command == "version":
        from . import __version__
        print(f"dead_simple_self_learning version {__version__}")
    else:
        parser.print_help()


def handle_embed(args):
    """Handle the embed command."""
    try:
        embedder = Embedder(model_name=args.model)
        embedding = embedder.embed(args.text)
        print(json.dumps(embedding))
    except Exception as e:
        print(f"Error generating embedding: {e}", file=sys.stderr)
        sys.exit(1)


def handle_enhance(args):
    """Handle the enhance command."""
    try:
        learner = SelfLearner(
            embedding_model=args.model,
            memory_path=args.memory,
            similarity_threshold=args.threshold
        )
        enhanced = learner.enhance_prompt(args.task, args.base_prompt)
        print(enhanced)
    except Exception as e:
        print(f"Error enhancing prompt: {e}", file=sys.stderr)
        sys.exit(1)


def handle_save(args):
    """Handle the save command."""
    try:
        learner = SelfLearner(
            embedding_model=args.model,
            memory_path=args.memory
        )
        learner.save_feedback(args.task, args.feedback)
        print(f"Feedback saved to {args.memory}")
    except Exception as e:
        print(f"Error saving feedback: {e}", file=sys.stderr)
        sys.exit(1)


def handle_view(args):
    """Handle the view command."""
    try:
        memory = Memory(file_path=args.memory)
        entries = memory.get_all()
        
        if args.format == "json":
            print(json.dumps(entries, indent=2))
        else:
            if not entries:
                print("Memory is empty")
            else:
                for i, entry in enumerate(entries):
                    print(f"\n{i+1}. Task: {entry['task']}")
                    print(f"   Feedback: {entry['feedback']}")
                    print(f"   Times used: {entry['times_used']}")
    except Exception as e:
        print(f"Error viewing memory: {e}", file=sys.stderr)
        sys.exit(1)


def handle_reset(args):
    """Handle the reset command."""
    try:
        if not args.confirm:
            confirm = input(f"Are you sure you want to reset memory at {args.memory}? [y/N] ")
            if confirm.lower() != "y":
                print("Reset cancelled")
                return
                
        memory = Memory(file_path=args.memory)
        memory.reset()
        print(f"Memory at {args.memory} has been reset")
    except Exception as e:
        print(f"Error resetting memory: {e}", file=sys.stderr)
        sys.exit(1)


def handle_export(args):
    """Handle the export command."""
    try:
        learner = SelfLearner(memory_path=args.memory)
        learner.export_memory(args.output)
        print(f"Memory exported to {args.output}")
    except Exception as e:
        print(f"Error exporting memory: {e}", file=sys.stderr)
        sys.exit(1)


def handle_import(args):
    """Handle the import command."""
    try:
        learner = SelfLearner(memory_path=args.memory)
        learner.import_memory(args.input)
        print(f"Memory imported from {args.input}")
    except Exception as e:
        print(f"Error importing memory: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 