import os
import sys

# Make sure you've installed the package: pip install -e .
try:
    from dead_simple_self_learning import SelfLearner
except ImportError:
    # Fallback for running the example without installing the package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dead_simple_self_learning import SelfLearner

# Initialize the self-learner
memory_path = "feedback_operations_memory.json"
learner = SelfLearner(
    memory_path=memory_path,
    clear_memory=True  # Start with fresh memory
)

def main():
    print("DEAD SIMPLE SELF-LEARNING - FEEDBACK OPERATIONS")
    
    # Add some sample feedback
    print("\nAdding sample feedback...")
    
    task1 = "Write a marketing email"
    feedback1 = "Keep it short and include a clear call to action"
    learner.save_feedback(task1, feedback1)
    
    task2 = "Write product description for headphones"
    feedback2 = "Highlight noise cancellation and battery life"
    learner.save_feedback(task2, feedback2)
    
    task3 = "Write an email to a customer about delayed shipping"
    feedback3 = "Apologize clearly and offer compensation"
    learner.save_feedback(task3, feedback3)
    
    # 1. List all feedback
    print("\n=== LISTING ALL FEEDBACK ===")
    learner.list_all_feedback(verbose=True)
    
    # 2. Find feedback for specific task
    print("\n=== FEEDBACK FOR SPECIFIC TASK ===")
    learner.list_feedback(task=task1)
    
    # 3. Find feedback containing substring
    print("\n=== FEEDBACK CONTAINING 'email' ===")
    learner.list_feedback_substring(task_substring="email")
    
    # 4. Export memory
    export_path = "exported_memory.json"
    print(f"\n=== EXPORTING MEMORY TO {export_path} ===")
    learner.export_memory(export_path)
    print(f"Memory exported to {export_path}")
    
    # 5. Remove feedback by index
    print("\n=== REMOVING FEEDBACK BY INDEX ===")
    learner.remove_feedback(index=1)
    learner.list_all_feedback()
    
    # 6. Remove feedback for specific task
    print("\n=== REMOVING FEEDBACK FOR SPECIFIC TASK ===")
    learner.remove_feedback_for_task(task=task3)
    learner.list_all_feedback()
    
    # 7. Reset memory
    print("\n=== RESETTING MEMORY ===")
    learner.reset_memory()
    print("Memory reset complete")
    learner.list_all_feedback()
    
    # 8. Import memory
    print(f"\n=== IMPORTING MEMORY FROM {export_path} ===")
    learner.import_memory(export_path)
    print(f"Memory imported from {export_path}")
    learner.list_all_feedback()
    
    # Clean up
    if os.path.exists(export_path):
        os.remove(export_path)
        print(f"\nRemoved {export_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 