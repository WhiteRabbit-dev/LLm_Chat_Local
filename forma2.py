import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tkinter import filedialog
import subprocess
import re
import sys
import threading
import time
import os

#Основные цвета формы:
bg_color = "bisque2"
bg_color_cons = "black"
weight_span = 3
color_span = "saddle brown"
bg_face = "bisque"
color_face = "green"
font_cons = "lime"
space_in = 5
space_out = 3

#системные
SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
LOGS_DIR = SCRIPT_DIR / "logs"
LOG_FILE = SCRIPT_DIR / "logs" / f"logs[{time.strftime("%Y-%m-%d %H.%M.%S")}].txt"
LLAMA_CPP_DIR = SCRIPT_DIR / "llama.cpp"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b4609/llama-b4609-bin-win-cuda-cu12.4-x64.zip"
LLAMA_CPP_URL2 = "https://github.com/ggml-org/llama.cpp/releases/download/b9181/cudart-llama-bin-win-cuda-12.4-x64.zip"
LLAMA_CPP_ZIP = SCRIPT_DIR / "llama_cpp_temp.zip"
LLAMA_CPP_ZIP2 = SCRIPT_DIR / "llama_cpp_temp2.zip"
SAVES_FILE = SCRIPT_DIR / "saves.txt"
FACES_DIR = SCRIPT_DIR / "faces"
FACE_STAND = FACES_DIR / "face_stand.txt"
FACE_TALK = FACES_DIR / "face_talk.txt"
FACE_NOW = FACE_STAND
#К самой модели
MODEL_PATH = ""  # модель Q4
CONTEXT_TOKENS_LIMIT = 500
users_name = "user"
models_name = "assistant"
chat_history = []
g_n_gpu_layers=int(-1)        # -1 = ВСЯ МОДЕЛЬ на GPU!
g_n_ctx=int(4096)             # Максимальный контекст (токенов)
g_n_threads=int(2)            # Потоки CPU для обработки
g_max_tokens=int(150)  # Максимум токенов в ответе
g_temperature=0.8  # "Креативность" модели (0.1 (строго по шаблону) - 2.0 (полный хаос в ответе))
g_min_p=0.05  #Минимальная вероятность слова при генерации (0.0 – 1.0)
g_top_k=int(30)  #Топ подходящих слов (лучшие по вероятностям)
g_top_p=0.95  # Выборка из подходящих (0.1-1.0)
g_repeat_penalty=1.15  # Штраф за повторы текста (-2.0 – 2.0)
g_frequency_penalty=0.6  #Штраф за частые слова (-2.0 – 2.0)
save_try = 0

llm = None #пересенная модели

def load_settings():
    if SAVES_FILE.exists():
        try:
            with open(SAVES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            global g_n_gpu_layers, g_n_ctx, g_n_threads, g_max_tokens
            global g_temperature, g_min_p, g_top_k, g_top_p, g_repeat_penalty, g_frequency_penalty
            global MODEL_PATH, save_try, CONTEXT_TOKENS_LIMIT, users_name, models_name
            for line in lines:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "n_gpu_layers":
                        if val: g_n_gpu_layers = int(val)
                        entry_n_gpu_layers.insert(0, g_n_gpu_layers)
                    elif key == "n_ctx":
                        if val: g_n_ctx = int(val)
                        entry_n_ctx.insert(0,g_n_ctx)
                    elif key == "n_threads":
                        if val: g_n_threads = int(val)
                        entry_n_threads.insert(0,g_n_threads)
                    elif key == "max_tokens":
                        if val: g_max_tokens = int(val)
                        entry_max_tokens.insert(0,g_max_tokens)
                    elif key == "temperature":
                        if val: g_temperature = float(val)
                        entry_temperature.insert(0,g_temperature)
                    elif key == "min_p":
                        if val: g_min_p = float(val)
                        entry_min_p.insert(0,g_min_p)
                    elif key == "top_k":
                        if val: g_top_k = int(val)
                        entry_top_k.insert(0,g_top_k)
                    elif key == "top_p":
                        if val: g_top_p = float(val)
                        entry_top_p.insert(0,g_top_p)
                    elif key == "repeat_penalty":
                        if val: g_repeat_penalty = float(val)
                        entry_repeat_penalty.insert(0,g_repeat_penalty)
                    elif key == "frequency_penalty":
                        if val: g_frequency_penalty = float(val)
                        entry_frequency_penalty.insert(0, g_frequency_penalty)
                    elif key == "CONTEXT_TOKENS_LIMIT":
                        if val: CONTEXT_TOKENS_LIMIT = float(val)
                        entry_CONTEXT_TOKENS_LIMIT.insert(0, CONTEXT_TOKENS_LIMIT)
                    elif key == "users_name":
                        if val: users_name = str(val)
                        entry_users_name.insert(0, users_name)
                    elif key == "models_name":
                        if val: models_name = str(val)
                        entry_models_name.insert(0, models_name)
                    elif key == "MODEL_PATH":
                        if val: MODEL_PATH = str(val)
                        models_file.insert(0, MODEL_PATH)
                    elif key == "prompt":
                        Console_prompt.insert("end", str(val))
            log_to_file("Настройки успешно загружены")
            path = Path(str(models_file.get()))
        except Exception as e:
            log_to_file(f"120.Ошибка чтения настроек: {e}", True)
    else:
        save_settings(True)

def save_settings(load = False):
    global save_try
    if load:
        save_try += 1
        try:
            with open(SAVES_FILE, "w", encoding="utf-8") as f:
                f.write(f"n_gpu_layers={g_n_gpu_layers}\n")
                f.write(f"n_ctx={g_n_ctx}\n")
                f.write(f"n_threads={g_n_threads}\n")
                f.write(f"max_tokens={g_max_tokens}\n")
                f.write(f"temperature={g_temperature}\n")
                f.write(f"min_p={g_min_p}\n")
                f.write(f"top_k={g_top_k}\n")
                f.write(f"top_p={g_top_p}\n")
                f.write(f"repeat_penalty={g_repeat_penalty}\n")
                f.write(f"frequency_penalty={g_frequency_penalty}\n")
                f.write(f"CONTEXT_TOKENS_LIMIT={CONTEXT_TOKENS_LIMIT}\n")
                f.write(f"users_name={users_name}\n")
                f.write(f"models_name={models_name}\n")
                f.write(f"MODEL_PATH={MODEL_PATH}\n")
                f.write(f"prompt={str(Console_prompt.get('1.0', 'end-1c'))}\n")
            log_to_file("Настройки сохранены")
        except Exception as e:
            log_to_file(f"147.не удалось сохранить настройки: {e}", True)
        load_settings()
    else:
        try:
            with open(SAVES_FILE, "w", encoding="utf-8") as f:
                f.write(f"n_gpu_layers={entry_n_gpu_layers.get()}\n")
                f.write(f"n_ctx={entry_n_ctx.get()}\n")
                f.write(f"n_threads={entry_n_threads.get()}\n")
                f.write(f"max_tokens={entry_max_tokens.get()}\n")
                f.write(f"temperature={entry_temperature.get()}\n")
                f.write(f"min_p={entry_min_p.get()}\n")
                f.write(f"top_k={entry_top_k.get()}\n")
                f.write(f"top_p={entry_top_p.get()}\n")
                f.write(f"repeat_penalty={entry_repeat_penalty.get()}\n")
                f.write(f"frequency_penalty={entry_frequency_penalty.get()}\n")
                f.write(f"CONTEXT_TOKENS_LIMIT={entry_CONTEXT_TOKENS_LIMIT.get()}\n")
                f.write(f"users_name={entry_users_name.get()}\n")
                f.write(f"models_name={entry_models_name.get()}\n")
                f.write(f"MODEL_PATH={models_file.get()}\n")
                f.write(f"prompt={str(Console_prompt.get('1.0', 'end-1c'))}\n")
            log_to_file("Настройки сохранены")
        except Exception as e:
            log_to_file(f"169.не удалось создать файл сохранения: {e}", True)

def handle_ctrl_key(event):
    kc = event.keycode
    #Ctrl+V (русская М) - Вставить
    if kc == 86:
        event.widget.event_generate('<<Paste>>')
        return 'break'
    #Ctrl+C (русская С) - Копировать
    elif kc == 67:
        event.widget.event_generate('<<Copy>>')
        return 'break'
    #Ctrl+X (русская Ч) - Вырезать
    elif kc == 88:
        event.widget.event_generate('<<Cut>>')
        return 'break'
    #Ctrl+Z (русская Я) - Отменить
    elif kc == 90:
        event.widget.event_generate('<<Undo>>')
        return 'break'

def activate_venv():
    global VENV_DIR
    if not VENV_PYTHON.exists():
        log_to_file("создание .venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        log_to_file(".venv создан")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    try:
        if sys.executable != str(VENV_PYTHON):
            log_to_file(f"Перезапуск в venv: {VENV_PYTHON}")
            #Передаём все аргументы и переменные окружения
            subprocess.Popen(
                [str(VENV_PYTHON)] + sys.argv,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            sys.exit(0)  # Выходим из текущего процесса
        else:
            log_to_file(f"venv уже прописан: {VENV_PYTHON}")
    except Exception as e:
        log_to_file(f"не удалось добавить директорию venv: {e}", True)
    try:
        import torch
        import llama_cpp
    except:
        install_torch_with_cuda()

    log_to_file("библиотеки подготовлены")

def install_cuda(index):
    subprocess.run(
        [str(VENV_PIP), "install", "torch",
         "--index-url", f"https://download.pytorch.org/whl/{index}"],
        check=True)
    log_to_file(f"PyTorch установлен версии {index}")

def install_torch_with_cuda():
    try:
        output = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True ).stdout
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if match:
            cuda_ver = match.group(1)
            log_to_file(f"Обнаружена CUDA: {cuda_ver}")
        else:
            log_to_file("CUDA не найдена, ставлю CPU-версию")
            cuda_ver = None
    except:
        log_to_file("nvidia-smi не сработал, ставлю CPU-версию")
        cuda_ver = None

    cuda_to_index = {
        "13.2": "cu124",
        "13.1": "cu124",
        "12.8": "cu124",
        "12.6": "cu124",
        "12.4": "cu124",
        "12.1": "cu121",
        "11.8": "cu118",
    }
    if cuda_ver :
        index = cuda_to_index.get(cuda_ver)
        log_to_file(f"пробуем установить PyTorch {index}...")
        try:
            install_cuda(index)
        except:
            try:
                log_to_file(f"при попытке установить PyTorch {index} что пошло не по плану...")
                log_to_file(f"пробуем установить (универсальную) PyTorch cu124...")
                index = "cu124"
                install_cuda(index)
            except Exception as e:
                log_to_file(f"261.не удалось установить PyTorch: {e}", True)
    else:
        log_to_file("Устанавливаю CPU-версию PyTorch")
        index = "cpu"
        install_cuda(index)
        log_to_file("Установлена CPU-версия PyTorch")
    install_Llama(index)

def install_Llama(index):
    if index == "cpu":
        try:
            log_to_file("llama-cpp-python устанавливаем (CPU-версию)")
            env = os.environ.copy()
            env["FORCE_CMAKE"] = "1"
            env["CMAKE_ARGS"] = "-DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off -DLLAMA_F16C=off"

            subprocess.run(
                [str(VENV_PIP), "install", "llama-cpp-python", "--extra-index-url", "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cpu", "--force-reinstall", "--no-cache-dir"],
                #[str(VENV_PIP), "install", "llama-cpp-python", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu", "--force-reinstall", "--no-cache-dir"],
                check=True, env=env
            )
            log_to_file("llama-cpp-python установлен (CPU-версия)")
        except Exception as e:
            log_to_file(f"279.не удалось установить llama-cpp-python: {e}", True)
    else:
        try:
            log_to_file(f"llama-cpp-python устанавливаем версию с поддержкой с CUDA {index}")
            subprocess.run(
                [str(VENV_PIP), "install", "llama-cpp-python",
                  "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{index}"],
                check=True
            )
            log_to_file(f"llama-cpp установлен с CUDA {index}")

        except Exception as e:
            log_to_file(f"291.не удалось установить llama-cpp-python: {e}", True)

def log_to_file(message, is_error=False):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    level = "ОШИБКА" if is_error else "ИНФО"
    log_line = f"[{timestamp}] {level}: {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())
    if is_error:
        on_closing()

def chk_syst():
    try:
        import torch
        print("Проверка GPU...")
        print(f"CUDA доступна:{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")
        print()
    except Exception as e:
        log_to_file(f"313.не удалось проверить видеокарту: {e}", True)

def get_llm():
    global llm
    global MODEL_PATH
    if MODEL_PATH == str(models_file.get()) and llm is not None:
        return llm
    else:
        unload_model()
        path = Path(str(models_file.get()))
        if path.exists() and path.is_file() and path.suffix == ".gguf":
            try:
                from llama_cpp import Llama
                llm = Llama(
                    model_path=str(models_file.get()),
                    n_gpu_layers=int(entry_n_gpu_layers.get()),
                    n_ctx=int(entry_n_ctx.get()),
                    n_threads=int(entry_n_threads.get()),
                    verbose=False,
                    use_mlock=False
                )
                return llm
            except Exception as e:
                log_to_file(f"335.Ошибка инициализации модели: {e}", True)
                return None
        else:
            log_to_file("ОШИБКА: неверный путь до модели, проверьте настройки.\n")
            return None

def unload_model():
    global llm
    if llm is not None:
        del llm
        llm = None
        import gc
        gc.collect()
        log_to_file("Модель выгружена из памяти")

def build_prompt(user_query, user):
    global chat_history,models_name
    chat_history.append({"role": user, "content": user_query})
    trim_history()
    models_name = str(entry_models_name.get())
    prompt = "<|im_start|>system: " + str(Console_prompt.get("1.0", "end-1c")) + "<|im_end|>\n"
    #print("=" *30)
    #print(f" История: {len(chat_history) // 2} диалогов")
    for i, msg in enumerate(chat_history, 1):
        role = msg["role"]
        #content_preview = msg["content"] + "..." if len(msg["content"]) > 50 else msg["content"]
        #print(f"   {i}. {role}: {content_preview}")
        prompt += f"<|im_start|>{role}: {msg['content']}<|im_end|>\n"
    prompt += f"<|im_start|>{models_name}:  "
    #add_text_cons("||||||||||||||||||" + prompt + "||||||||||||||||||" )
    #total_tokens = count_tokens(prompt)
    #print(f" ПРОМПТ: {total_tokens} токенов ({total_tokens / 4096 * 100:.0f}% контекста)")
    #print("=" * 30)
    return prompt

def enter_btn():
    try:
        global FACE_NOW,FACE_TALK,FACE_STAND
        FACE_NOW=FACE_TALK
        model = get_llm()
        if model is not None:
            global chat_history
            query = Enter_cons.get()
            Enter_cons.delete(0, "end")
            add_text_cons(query+"\n")
            start_time = time.time()
            prompt = build_prompt(query, str(entry_users_name.get()))
            #add_text_cons("Промпт:" + prompt + "\n")
            response = llm(
                prompt,
                max_tokens = int(entry_max_tokens.get()),  # Максимум токенов в ответе
                temperature = float(entry_temperature.get()),  # "Креативность" модели (0.1 (строго по шаблону) - 2.0 (полный хаос в ответе))
                min_p = float(entry_min_p.get()),  #Минимальная вероятность слова при генерации (0.0 – 1.0)
                top_k = int(entry_top_k.get()),  #Топ подходящих слов (лучшие по вероятностям)
                top_p = float(entry_top_p.get()),  # Выборка из подходящих (0.1-1.0)
                repeat_penalty = float(entry_repeat_penalty.get()),  # Штраф за повторы текста (-2.0 – 2.0)
                frequency_penalty = float(entry_frequency_penalty.get()),  # Штраф за частые слова (-2.0 – 2.0)
                stop=["<|im_end|>"]  # Стоп-слова
            )
            answer = response['choices'][0]['text'].strip()
            elapsed = time.time() - start_time
            chat_history.append({"role": models_name, "content": answer})
            #add_text_cons(f"{elapsed:.1f}с | Токенов в ответе: ~{count_tokens(answer)}" + "\n")
            lable_stat_1.config(text=f"{elapsed:.1f}с | Токенов в ответе: ~{count_tokens(answer)}")
            add_text_cons(f"{models_name}: {answer}" + "\n")
            add_text_cons (">")
        else:
            add_text_cons("ОШИБКА: модель не загружена, подробнее см. в логах\n")
            return
        root.after(1000, default_face)
    except Exception as e:
        add_text_cons(f"ОШИБКА: {e}" + "\n")
        log_to_file(f"ОШИБКА: {e}" + "\n")
        add_text_cons(" Продолжаем..." + "\n")
        log_to_file(" Продолжаем...")
        root.after(1500, default_face)

def default_face():
    global FACE_NOW,FACE_STAND
    FACE_NOW = FACE_STAND

def count_tokens(text):
    #Подсчитывает токены в тексте (примерно)
    return len(text) // 4

def trim_history():
    global chat_history
    current_tokens = sum(count_tokens(msg["content"]) for msg in chat_history) + count_tokens(str(Console_prompt.get("1.0", "end-1c")))
    while len(chat_history) > 0 and current_tokens > CONTEXT_TOKENS_LIMIT:
        removed = chat_history.pop(0)
        removed2 = chat_history.pop(0)# Удаляем самую старую пару сообщений
        current_tokens -= count_tokens(removed["content"]+removed2["content"])
    # print(f"    Обрезано до {len(chat_history)} сообщений ({current_tokens} токенов)")

def add_text_cons(text):
    Console_main.config(state="normal")
    Console_main.insert("end", text)
    Console_main.see("end")
    Console_main.config(state="disabled")

def browse_model_file():
    filename = filedialog.askopenfilename(
        title="Выберите файл модели",
        filetypes=[
            ("GGUF модели", "*.gguf"),
            ("Все файлы", "*.*")
        ],
        initialdir=str(SCRIPT_DIR)  # Начать с папки проекта
    )
    if filename:
        models_file.delete(0, "end")
        models_file.insert(0, filename)

def make_tooltip(widget, text):
    tip = None
    def show(e):
        nonlocal tip
        tip = tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{e.x_root+15}+{e.y_root+10}")
        tk.Label(tip, text=text, bg=bg_color, highlightbackground=color_span, highlightthickness=weight_span).pack()
    def hide(e):
        nonlocal tip
        if tip: tip.destroy(); tip = None
    widget.bind('<Enter>', show)
    widget.bind('<Leave>', hide)

"""def enter_btn():
    text = Enter_cons.get()
    if text.strip():
        Console_main.config(state="normal")
        Console_main.insert("end", text + "\n")
        Console_main.config(state="disabled")
        Console_main.see("end")
        Enter_cons.delete(0, "end")
        # Выполняем команду в консоли
        try:
                # Запускаем команду и получаем результат
            result = subprocess.run(
                text,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(SCRIPT_DIR)  # выполнять в папке проекта
            )
            Console_main.config(state="normal")
            # Выводим результат
            if result.stdout:
                Console_main.insert("end", result.stdout)
            if result.stderr:
                Console_main.insert("end", result.stderr)
            Console_main.insert("end", "> ")  # новый промпт
            Console_main.config(state="disabled")
            Console_main.see("end")
        except Exception as e:
            Console_main.config(state="normal")
            Console_main.insert("end", f"Ошибка: {e}\n> ")
            Console_main.config(state="disabled")
            Console_main.see("end")"""

"""СОЗДАНИЕ САМОЙ ФОРМЫ """
root = tk.Tk()
root.bind_class('all', '<Control-KeyPress>', handle_ctrl_key)
root.title("ФОРМА V2")
root.geometry("700x438")
root.minsize(700, 438)
root.resizable(True, True)

notebook = ttk.Notebook(root)  #панель вкладок
notebook.pack(fill='both', expand=True)
style = ttk.Style()
style.configure('TNotebook', background=bg_color)
style.configure('TFrame', background=bg_color)

# region === вкладка 1 ===
# ВКЛАДКА 1
main_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="Чат")

main_tab.grid_columnconfigure(0, weight=2)
main_tab.grid_columnconfigure(1, weight=1)
main_tab.grid_rowconfigure(0, weight=1)

# Блок 1: слева 2/3
# , relief='solid', bd=1
block1 = tk.LabelFrame(main_tab,text="Чат",bd=0, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block1.grid_propagate(False)
block1.grid(row=0, column=0, sticky='nsew', padx=space_out, pady=space_out)
block1.grid_rowconfigure(0, weight=1)
block1.grid_rowconfigure(1, weight=0)
block1.grid_columnconfigure(0, weight=1)
block1.grid_columnconfigure(1, weight=0)

#типа консоль
Console_main = tk.Text(block1,state="disabled", bg=bg_color_cons, fg=font_cons, width=1, height=1)
Console_main.config(state="normal")
Console_main.insert("end", "> ")  # с новой строки
Console_main.config(state="disabled")
Console_main.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=space_in, pady=(space_in,0))

Enter_cons = tk.Entry(block1, bg="white")
Enter_cons.grid(row=1, column=0, sticky="nsew", padx=(space_in,0), pady=(0,space_in))
Enter_cons.bind("<Return>", lambda e: enter_btn())
Btn_enter = tk.Button(block1, text="Ввод")
Btn_enter.grid(row=1, column=1, sticky="nsew", padx=(0,space_in), pady=(0,space_in))
Btn_enter.config(command=enter_btn)

# Правая колонка: ПРАВО 1/3
#, relief='solid', bd=1
right_column = tk.Frame(main_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
right_column.grid(row=0, column=1, sticky='nsew', padx=space_out, pady=space_out)
right_column.grid_rowconfigure(0, weight=1)
right_column.grid_rowconfigure(1, weight=1)
right_column.grid_columnconfigure(0, weight=1)

# Блок 2: ВЕРХ 2/3
# , relief='solid', bd=2
block2 = tk.LabelFrame(right_column,text="Промпт",bd=0,bg=bg_color)
block2.grid_propagate(False)
block2.grid(row=0, column=0, sticky='nsew', padx=space_in, pady=space_in)

block2.grid_rowconfigure(0, weight=4)
block2.grid_rowconfigure(1, weight=1)
block2.grid_rowconfigure(2, weight=1)
block2.grid_columnconfigure(0, weight=1)
block2.grid_columnconfigure(1, weight=1)

Console_prompt = tk.Text(block2, bg=bg_color_cons, fg=font_cons, width=1, height=1)
Console_prompt.grid(row=0, column=0, columnspan=2, sticky="nsew")

lable_stat_1 = tk.Label(block2, bg=bg_color)
lable_stat_1.grid(row=1, column=0, sticky="nw")
#
#
#
# Блок 3: НИЗ 1/3
# , relief='solid', bd=2
block3 = tk.Frame(right_column, bg=bg_color)
block3.grid_propagate(False)
block3.grid(row=1, column=0, sticky='nsew', padx=space_in, pady=space_in)
block3.grid_rowconfigure(0, weight=1)
block3.grid_columnconfigure(0, weight=1)
label_face = tk.Label(block3,justify="left", fg=font_cons,bg=bg_color_cons)
label_face.grid(row=0, column=0, sticky="nsew")

def resize_label(event):
    text=(label_face.cget("text"))
    num_lines= text.count("\n")+1
    if num_lines > 0:
        lines = text.split("\n")
        max_line_len = max(len(line) for line in lines)
        pixel_on_row = (root.winfo_width() / 3 )// max_line_len
        pixel_on_line = (root.winfo_height() / 2 ) // num_lines
        font_size = min (int(pixel_on_line  // 1.33) , int(pixel_on_row // 0.9))
        label_face.config(font=("Courier", font_size))
block3.bind("<Configure>", resize_label)
# endregion

# region === вкладка 2 ===
# ВКЛАДКА 2
second_tab = ttk.Frame(notebook)
notebook.add(second_tab, text="Настройки")

second_tab.grid_rowconfigure(0, weight=0)
second_tab.grid_rowconfigure(1, weight=1)
second_tab.grid_columnconfigure(0, weight=1)
second_tab.grid_columnconfigure(1, weight=1)

block4 = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block4.grid(row=0, column=0,columnspan=2, sticky="nsew", padx=space_out, pady=space_out)
block4.grid_rowconfigure(0, weight=1)
block4.grid_columnconfigure(0, weight=1)
block4.grid_columnconfigure(1, weight=0)

models_file = tk.Entry(block4, bg="white")
models_file.grid(row=0, column=0, sticky="nsew", padx=(space_in,0), pady=space_in)
Btn_browse = tk.Button(block4, text="Обзор", command=browse_model_file)
Btn_browse.grid(row=0, column=1, sticky="nsew", padx=(0, space_in), pady=space_in)

block5 = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span,bg=bg_color)
block5.grid_propagate(False)
block5.grid(row=1, column=0, sticky="nsew", padx=space_out, pady=space_out)
block5.grid_rowconfigure(0, weight=0)
block5.grid_rowconfigure(1, weight=0)
block5.grid_rowconfigure(2, weight=0)
block5.grid_rowconfigure(3, weight=0)
block5.grid_rowconfigure(4, weight=0)
block5.grid_rowconfigure(5, weight=0)
block5.grid_rowconfigure(6, weight=0)
block5.grid_rowconfigure(7, weight=0)
block5.grid_rowconfigure(8, weight=0)
block5.grid_rowconfigure(9, weight=0)
block5.grid_rowconfigure(10, weight=0)
block5.grid_rowconfigure(11, weight=0)
block5.grid_rowconfigure(12, weight=0)
block5.grid_rowconfigure(13, weight=0)
block5.grid_rowconfigure(14, weight=0)
block5.grid_columnconfigure(0, weight=1)
block5.grid_columnconfigure(1, weight=1)
block5.grid_columnconfigure(2, weight=2)

entr_width = 5
tk.Label(block5, text="Настройки Llama:", bg=bg_color).grid(row=0, column=0, columnspan=3, sticky="nsew")
tk.Label(block5, text="n_gpu_layers:", bg=bg_color).grid(row=1, column=0, sticky="nsew")
entry_n_gpu_layers = tk.Entry(block5, bg="white", width=entr_width)
entry_n_gpu_layers.grid(row=1, column=1, sticky="ns")
tk.Label(block5, text="n_ctx:", bg=bg_color).grid(row=2, column=0, sticky="nsew")
entry_n_ctx = tk.Entry(block5, bg="white", width=entr_width)
entry_n_ctx.grid(row=2, column=1, sticky="ns")
tk.Label(block5, text="n_threads:", bg=bg_color).grid(row=3, column=0, sticky="nsew")
entry_n_threads = tk.Entry(block5, bg="white", width=entr_width)
entry_n_threads.grid(row=3, column=1, sticky="ns")
tk.Label(block5, text="max_tokens:", bg=bg_color).grid(row=4, column=0, sticky="nsew")
entry_max_tokens = tk.Entry(block5, bg="white", width=entr_width)
entry_max_tokens.grid(row=4, column=1, sticky="ns")
tk.Label(block5, text="temperature:", bg=bg_color).grid(row=5, column=0, sticky="nsew")
entry_temperature = tk.Entry(block5, bg="white", width=entr_width)
entry_temperature.grid(row=5, column=1, sticky="ns")
tk.Label(block5, text="min_p:", bg=bg_color).grid(row=6, column=0, sticky="nsew")
entry_min_p = tk.Entry(block5, bg="white", width=entr_width)
entry_min_p.grid(row=6, column=1, sticky="ns")
tk.Label(block5, text="top_k:", bg=bg_color).grid(row=7, column=0, sticky="nsew")
entry_top_k = tk.Entry(block5, bg="white", width=entr_width)
entry_top_k.grid(row=7, column=1, sticky="ns")
tk.Label(block5, text="top_p:", bg=bg_color).grid(row=8, column=0, sticky="nsew")
entry_top_p = tk.Entry(block5, bg="white", width=entr_width)
entry_top_p.grid(row=8, column=1, sticky="ns")
tk.Label(block5, text="repeat_penalty:", bg=bg_color).grid(row=9, column=0, sticky="nsew")
entry_repeat_penalty = tk.Entry(block5, bg="white", width=entr_width)
entry_repeat_penalty.grid(row=9, column=1, sticky="ns")
tk.Label(block5, text="frequency_penalty:", bg=bg_color).grid(row=10, column=0, sticky="nsew")
entry_frequency_penalty = tk.Entry(block5, bg="white", width=entr_width)
entry_frequency_penalty.grid(row=10, column=1, sticky="ns")
tk.Label(block5, text="память ии (токены):", bg=bg_color).grid(row=11, column=0, sticky="nsew")
entry_CONTEXT_TOKENS_LIMIT = tk.Entry(block5, bg="white", width=entr_width)
entry_CONTEXT_TOKENS_LIMIT.grid(row=11, column=1, sticky="ns")
tk.Label(block5, text="Имя пользователя для чата:", bg=bg_color).grid(row=12, column=0, sticky="nsew")
entry_users_name = tk.Entry(block5, bg="white", width=entr_width)
entry_users_name.grid(row=12, column=1, sticky="nsew")
tk.Label(block5, text="Имя модели для чата:", bg=bg_color).grid(row=13, column=0, sticky="nsew")
entry_models_name = tk.Entry(block5, bg="white", width=entr_width)
entry_models_name.grid(row=13, column=1, sticky="nsew")

btn_save = tk.Button(block5, text="Ручное сохранение настроек в файл", command=save_settings)
btn_save.grid(row=14, column=0, columnspan=3, sticky="nsew")

make_tooltip(entry_n_gpu_layers, "количество слоев которые будут выгружены на видеокарту \n(при ее наличии) (-1 = ВСЯ МОДЕЛЬ на GPU!).")
make_tooltip(entry_n_ctx, "Максимальный контекст (токенов)")
make_tooltip(entry_n_threads, "Потоки CPU для обработки")
make_tooltip(entry_max_tokens, "Максимум токенов в ответе")
make_tooltip(entry_temperature, '"Креативность" модели (0.1 (строго по шаблону) - 2.0 (полный хаос в ответе)')
make_tooltip(entry_min_p, "Минимальная вероятность слова при генерации (0.0 – 1.0)")
make_tooltip(entry_top_k, "Топ подходящих слов (лучшие по вероятностям")
make_tooltip(entry_top_p, "Выборка из подходящих (0.1-1.0)")
make_tooltip(entry_repeat_penalty, "Штраф за повторы текста (-2.0 – 2.0)")
make_tooltip(entry_frequency_penalty, "Штраф за частые слова (-2.0 – 2.0)")
make_tooltip(entry_CONTEXT_TOKENS_LIMIT, 'Количество токенов отведенное ей под "память". \nЧасть диалога  которые в влезут в ти токены будет выгружена в промпт \nпри последующих запросах.')
make_tooltip(entry_users_name, "Имя пользователя нужно только для чата и лучшего сопоставления \nнейросетью имени с текущим пользователем в чате.")
make_tooltip(entry_models_name, "Имя модели нужно для лучшего сопоставления нейросети с персонажем \nкоторого она отыгрывает. (из промпта выковыривать гемор)")
make_tooltip(btn_save, "При сохранении будут записаны все текущие настройки \nв том числе и текущий промпт.")


block6 = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block6.grid_propagate(False)
block6.grid(row=1, column=1, sticky="nsew", padx=space_out, pady=space_out)

block6.grid_rowconfigure(0, weight=1)
block6.grid_rowconfigure(1, weight=0)
block6.grid_columnconfigure(0, weight=1)

def donation():
    try:
        import webbrowser
        webbrowser.open("https://www.donationalerts.com/r/whiterabbit_")
    except Exception as e:
        log_to_file(f"717.Ошибка открытия браузера: {e}")

tk.Label(block6, text="Автор старался!\nАвтор хороший!\nАвтора можно поддержать!\n\n\n\nПозднее этот блок будет использован под \nболее важные вещи но пока так))", bg=bg_color).grid(row=0, column=0, sticky="nsew")
btn_save = tk.Button(block6, text="Поддержать автора!", command=donation)
btn_save.grid(row=1, column=0, sticky="ew")
# endregion

def on_closing():
    save_settings()
    log_to_file("Приложение закрыто")
    root.destroy()

def thread_animation_loop():
    try:
        global FACE_NOW
        face_now = str(FACE_NOW)
        face = open(face_now, "r", encoding="utf-8").read()

        ch = -1
        line = 0
        while True:
            if not face_now == FACE_NOW:
                face = open(FACE_NOW, "r", encoding="utf-8").read()
                face_now = FACE_NOW
            if line == 0 or line == 6:
                ch *= -1

            if line == 0:
                root.after(0, lambda: label_face.config(text=face))
            if line == 1:
                root.after(0, lambda: label_face.config(text=face))
            elif line == 2:
                root.after(0, lambda: label_face.config(text="\n" + face))
            elif line == 3:
                root.after(0, lambda: label_face.config(text="\n\n" + face))
            elif line == 4:
                root.after(0, lambda: label_face.config(text="\n\n\n" + face))
            elif line == 5:
                root.after(0, lambda: label_face.config(text="\n\n\n\n" + face))
            elif line == 6:
                root.after(0, lambda: label_face.config(text="\n\n\n\n" + face))

            # lines.insert(0, "# Настройки программы\n")
            line += ch
            time.sleep(0.3)
    except Exception as e:
        log_to_file(f"Произошла ошибка при отрисовке морды: {e}")

root.after(100, lambda: threading.Thread(target=thread_animation_loop, daemon=True).start())

root.protocol("WM_DELETE_WINDOW", on_closing)
log_to_file("Приложение запущено")

activate_venv()
chk_syst()
load_settings()

log_to_file("Запуск GUI")
root.mainloop()