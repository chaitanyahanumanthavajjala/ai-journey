import json

def add_task():
    task = input("Enter a task: ")
    if task.strip() == "":
        print("Task cannot be empty.")
        return
    task_dic = {"task": task.strip(), "completed": False}
    tasks.append(task_dic)
    print(f"Added: {task.strip()} to task list.")
    save_tasks()
    
def view_tasks():
    if not tasks:
        print("No tasks yet.")
    else:
        for i, task in enumerate(tasks, start=1):
            status = "[X]" if task["completed"] else "[ ]"
            print(f"{i}. {task['task']}  {status}")
        print(f"Total tasks: {len(tasks)}")

def mark_complete():
    task_number = input("Enter the task number to mark as complete: ")
    if not tasks:
        print("No tasks to mark as complete.")
        return
    try:
        task_index = int(task_number) - 1
        if 0 <= task_index < len(tasks):
            tasks[task_index]["completed"] = True
            print(f"Marked as complete: {tasks[task_index]['task']}")
            save_tasks()
        else:
            print("Invalid task number.")
    except ValueError:
        print("Please enter a valid task number.")

def delete_task():
    task_number = input("Enter the task number to delete: ")
    if not tasks:
        print("No tasks to delete.")
        return
    try:
        task_index = int(task_number) - 1
        if 0 <= task_index < len(tasks):
            print(f"Deleted: {tasks[task_index]['task']}")
            del tasks[task_index]
            save_tasks()
        else:
            print("Invalid task number.")
    except ValueError:
        print("Please enter a valid task number.")

def save_tasks():
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)

def load_tasks():
    global tasks
    try:
        with open("tasks.json", "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        tasks = []

def main():
    load_tasks()
    while True:
        print("\n1. Add task\n2. View tasks\n3. Mark task as complete\n4. Delete task\n5. Quit")
        choice = input("Choose an option: ")
        if choice == "1":
            add_task()
        elif choice == "2":
            view_tasks()
        elif choice == "3":
            mark_complete()
        elif choice == "4":
            delete_task()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Not a valid option, try again.")

main()