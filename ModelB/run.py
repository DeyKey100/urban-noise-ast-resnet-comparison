import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# Добавляем папку со скриптом в пути для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import predict_sound
except ImportError:
    messagebox.showerror("Ошибка", "Файл predict.py не найден в той же папке!")
    sys.exit(1)


def main():
    root = tk.Tk()
    root.withdraw()

    initial_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sounds")
    if not os.path.exists(initial_dir):
        initial_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = filedialog.askopenfilename(
        title="Выберите аудиофайл для оценки",
        initialdir=initial_dir,
        filetypes=[("WAV файлы", "*.wav"), ("Все файлы", "*.*")]
    )

    if file_path:
        print(f"Выбран файл: {file_path}")
        print("Запуск анализа...\n")
        try:
            predict_sound(file_path)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

        input("\nНажмите Enter, чтобы закрыть окно...")
    else:
        print("Файл не выбран. Выход.")


if __name__ == "__main__":
    main()