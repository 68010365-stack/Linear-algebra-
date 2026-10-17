import csv
import numpy as np  # <--- เพิ่มเข้ามาสำหรับคำนวณ Matrix
import math
from multiprocessing import Pool, cpu_count
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os


# ==========================================
# 1. ฟังก์ชันคำนวณทางคณิตศาสตร์ (เขียนเองทั้งหมด)
# ==========================================
def cosine_similarity(vec_a, vec_b):
  dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
  magnitude_a = math.sqrt(sum(a**2 for a in vec_a))
  magnitude_b = math.sqrt(sum(b**2 for b in vec_b))

  if magnitude_a == 0 or magnitude_b == 0:
    return 0.0
  return dot_product / (magnitude_a * magnitude_b)


def euclidean_distance(vec_a, vec_b):
  squared_diff_sum = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
  return math.sqrt(squared_diff_sum)


# ==========================================
# 2. ฟังก์ชันโหลด Dataset จากไฟล์ CSV
# ==========================================
def load_dataset_matrix_ready(file_path):
  dataset = []
  with open(file_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
      dataset.append({
          "food_name": row["food_name"],
          "ingredients": [
              ing.strip().lower() for ing in row["ingredients"].split(",")
          ],
          "nutrients": [
              float(row["calories_per_serving"]),
              float(row["protein_g"]),
              float(row["carbohydrates_g"]),
              float(row["fat_g"]),
          ],
      })
  return dataset


def worker_process(
    chunk_data,
    all_ingredients,
    fridge_ingredients,
    fridge_vector,
    target_nutrients,
):
  local_results = []
  for food in chunk_data:
    recipe_ingredients = set(food["ingredients"])
    fridge_set = set(fridge_ingredients)
    common_ingredients = recipe_ingredients.intersection(fridge_set)

    # กรองเกณฑ์ 10% ขึ้นไป (ปรับเปลี่ยนได้ตามต้องการ)
    if len(recipe_ingredients) == 0:
      continue
    match_percentage = (len(common_ingredients) / len(recipe_ingredients)) * 100
    if match_percentage < 10:
      continue

    food_vector = [
        1 if item in food["ingredients"] else 0 for item in all_ingredients
    ]
    cos_sim = cosine_similarity(fridge_vector, food_vector)
    euc_dist = euclidean_distance(target_nutrients, food["nutrients"])

    local_results.append((food, cos_sim, euc_dist))
  return local_results


# ==========================================
# 3. ฟังก์ชันหลักสำหรับแนะนำอาหาร
# ==========================================
def recommend_food_matrix(csv_path, fridge_ingredients, target_nutrients):
  dataset = load_dataset_matrix_ready(csv_path)
  if not dataset:
    return []

  # สร้าง Feature Space (Vocabulary ของวัตถุดิบทั้งหมดในระบบ)
  all_ingredients = sorted(
      list(set(ing for item in dataset for ing in item["ingredients"]))
  )
  num_foods = len(dataset)
  num_ingredients = len(all_ingredients)

  # แปลง Ingredient Dataset ให้เป็น 2D Matrix ขนาด (M x N)
  ingredient_matrix = np.zeros((num_foods, num_ingredients))
  for i, food in enumerate(dataset):
    for ing in food["ingredients"]:
      if ing in all_ingredients:
        j = all_ingredients.index(ing)
        ingredient_matrix[i, j] = 1.0

  # แปลง Fridge Ingredients ให้เป็น Vector ขนาด (1 x N)
  fridge_vector = np.zeros(num_ingredients)
  for ing in fridge_ingredients:
    if ing in all_ingredients:
      j = all_ingredients.index(ing)
      fridge_vector[j] = 1.0

  # คำนวณเปอร์เซ็นต์วัตถุดิบที่ตรงกัน (>= 10%) แบบ Vectorization
  recipe_lengths = np.sum(ingredient_matrix, axis=1)
  common_counts = np.dot(ingredient_matrix, fridge_vector)
  recipe_lengths_safe = np.where(recipe_lengths == 0, 1, recipe_lengths)
  match_percentages = (common_counts / recipe_lengths_safe) * 100

  valid_indices = np.where(match_percentages >= 10)[0]
  if len(valid_indices) == 0:
    return []

  valid_ingredient_matrix = ingredient_matrix[valid_indices]
  valid_dataset = [dataset[i] for i in valid_indices]

  # คำนวณ Cosine Similarity ด้วย Matrix Dot Product
  dot_products = np.dot(valid_ingredient_matrix, fridge_vector)
  matrix_norms = np.linalg.norm(valid_ingredient_matrix, axis=1)
  fridge_norm = np.linalg.norm(fridge_vector)

  denominator = matrix_norms * fridge_norm
  cosine_sims = np.where(denominator == 0, 0.0, dot_products / denominator)

  # สร้าง Nutrient Matrix และคำนวณ Euclidean Distance ด้วย Matrix Broadcasting ($L_2$ Norm)
  nutrient_matrix = np.array([food["nutrients"] for food in valid_dataset])
  target_vector = np.array(target_nutrients)

  euclidean_dists = np.linalg.norm(nutrient_matrix - target_vector, axis=1)

  # Normalize ระยะทางและรวมคะแนน Final Score
  max_euc = np.max(euclidean_dists)
  if max_euc == 0:
    euc_scores = np.ones_like(euclidean_dists)
  else:
    euc_scores = 1.0 - (euclidean_dists / max_euc)

  final_scores = (0.5 * cosine_sims) + (0.5 * euc_scores)

  # จัดเรียงลำดับผลลัพธ์จากคะแนนมากไปน้อย
  sorted_order = np.argsort(final_scores)[::-1]

  scored_results = []
  for idx in sorted_order:
    scored_results.append({
        "food_name": valid_dataset[idx]["food_name"],
        "ingredients": valid_dataset[idx]["ingredients"],
        "nutrients": valid_dataset[idx]["nutrients"],
        "cosine_similarity": float(cosine_sims[idx]),
        "euclidean_distance": float(euclidean_dists[idx]),
        "final_score": float(final_scores[idx]),
    })

  return scored_results


# ==========================================
# 4. ส่วนติดต่อผู้ใช้งาน (Tkinter UI)
# ==========================================
class FoodRecommenderApp:

  def __init__(self, root):
    self.root = root
    self.root.title("ระบบแนะนำอาหารตามวัตถุดิบและสารอาหาร")
    self.root.geometry("900x650")

    self.fridge_list = []
    self.csv_path = tk.StringVar(
        value="dataset.csv"
    )  # ค่าเริ่มต้น หรือเปลี่ยนตาม path เครื่อง
    self.csv_path = tk.StringVar(value="dataset.csv")
    self.config_file = "app_config.json"  # ไฟล์สำหรับเก็บบันทึกข้อมูล
    self.create_widgets()

  def create_widgets(self):
    # --- Frame เลือกไฟล์ Dataset ---
    file_frame = tk.LabelFrame(
        self.root, text=" ตั้งค่าไฟล์ Dataset ", font=("Arial", 11, "bold")
    )
    file_frame.pack(fill="x", padx=10, pady=5)

    tk.Label(file_frame, text="ไฟล์ CSV:", font=("Arial", 10)).pack(
        side="left", padx=5, pady=5
    )
    tk.Entry(
        file_frame, textvariable=self.csv_path, width=60, font=("Arial", 10)
    ).pack(side="left", padx=5, pady=5)
    tk.Button(
        file_frame,
        text="เลือกไฟล์",
        command=self.browse_file,
        font=("Arial", 9),
    ).pack(side="left", padx=5, pady=5)

    # --- Main Content Frame (แบ่งซ้าย-ขวา) ---
    main_frame = tk.Frame(self.root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # --- ฝั่งซ้าย: วัตถุดิบในตู้เย็น ---
    left_frame = tk.LabelFrame(
        main_frame,
        text=" วัตถุดิบที่มีในตู้เย็น ",
        font=("Arial", 11, "bold"),
    )
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    tk.Label(
        left_frame,
        text="พิมพ์ชื่อวัตถุดิบแล้วกด Enter:",
        font=("Arial", 10),
    ).pack(anchor="w", padx=5, pady=2)
    self.ingredient_entry = tk.Entry(left_frame, font=("Arial", 11))
    self.ingredient_entry.pack(fill="x", padx=5, pady=5)
    self.ingredient_entry.bind("<Return>", self.add_ingredient)

    tk.Label(
        left_frame, text="รายการวัตถุดิบในตู้เย็น:", font=("Arial", 10)
    ).pack(anchor="w", padx=5, pady=2)

    # Listbox แสดงรายการวัตถุดิบ พร้อม Scrollbar
    list_container = tk.Frame(left_frame)
    list_container.pack(fill="both", expand=True, padx=5, pady=5)

    self.listbox = tk.Listbox(list_container, font=("Arial", 10))
    self.listbox.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(
        list_container, orient="vertical", command=self.listbox.yview
    )
    scrollbar.pack(side="right", fill="y")
    self.listbox.config(yscrollcommand=scrollbar.set)

    tk.Button(
        left_frame,
        text="ลบรายการที่เลือก",
        command=self.remove_ingredient,
        bg="#ff9999",
        font=("Arial", 9),
    ).pack(fill="x", padx=5, pady=5)

    # --- ฝั่งขวา: สารอาหารเป้าหมาย & ผลลัพธ์ ---
    right_frame = tk.Frame(main_frame)
    right_frame.pack(
        side="right", fill="both", expand=True, padx=5, pady=5
    )

    # ช่องกรอกสารอาหาร 4 ช่อง
    nutrient_frame = tk.LabelFrame(
        right_frame,
        text=" สารอาหารที่ต้องการ (Target Nutrients) ",
        font=("Arial", 11, "bold"),
    )
    nutrient_frame.pack(fill="x", padx=0, pady=0)

    fields = [
        ("Calories (kcal):", "150"),
        ("Protein (g):", "3"),
        ("Carbs (g):", "27"),
        ("Fat (g):", "4"),
    ]
    self.nutrient_entries = {}

    for idx, (label_text, default_val) in enumerate(fields):
      tk.Label(nutrient_frame, text=label_text, font=("Arial", 10)).grid(
          row=idx, column=0, sticky="w", padx=5, pady=3
      )
      ent = tk.Entry(nutrient_frame, font=("Arial", 10), width=15)
      ent.insert(0, default_val)
      ent.grid(row=idx, column=1, sticky="ew", padx=5, pady=3)
      self.nutrient_entries[label_text] = ent

    # ปุ่มกดค้นหา
    tk.Button(
        right_frame,
        text="🔍 ค้นหาเมนูอาหารแนะนำ",
        command=self.run_recommendation,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 11, "bold"),
    ).pack(fill="x", padx=0, pady=10)

    # แสดงผลลัพธ์
    result_frame = tk.LabelFrame(
        right_frame,
        text=" ผลลัพธ์การแนะนำอาหาร ",
        font=("Arial", 11, "bold"),
    )
    result_frame.pack(fill="both", expand=True, padx=0, pady=0)

    self.result_text = tk.Text(
        result_frame, wrap="word", font=("Arial", 10), height=15
    )
    self.result_text.pack(side="left", fill="both", expand=True)
    res_scroll = tk.Scrollbar(
        result_frame, orient="vertical", command=self.result_text.yview
    )
    res_scroll.pack(side="right", fill="y")
    self.result_text.config(yscrollcommand=res_scroll.set)

  def browse_file(self):
    filename = filedialog.askopenfilename(
        title="เลือกไฟล์ Dataset CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if filename:
      self.csv_path.set(filename)

  def add_ingredient(self, event):
    item = self.ingredient_entry.get().strip().lower()
    if item:
      if item not in self.fridge_list:
        self.fridge_list.append(item)
        self.listbox.insert(tk.END, item)
      self.ingredient_entry.delete(0, tk.END)

  def remove_ingredient(self):
    selected_indices = self.listbox.curselection()
    if selected_indices:
      index = selected_indices[0]
      item = self.listbox.get(index)
      self.listbox.delete(index)
      if item in self.fridge_list:
        self.fridge_list.remove(item)

  # ==========================================
  # ฟังก์ชันบันทึกและโหลดข้อมูล (เพิ่มใหม่ตรงนี้)
  # ==========================================
  def save_data(self):
    """บันทึกข้อมูลปัจจุบันลงในไฟล์ JSON ก่อนปิดโปรแกรม"""
    nutrients_data = {}
    for label, ent in self.nutrient_entries.items():
      nutrients_data[label] = ent.get()

    data_to_save = {
        "csv_path": self.csv_path.get(),
        "fridge_list": self.fridge_list,
        "nutrients": nutrients_data,
    }
    try:
      with open(self.config_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
      print(f"Error saving config: {e}")

  def load_data(self):
    """โหลดข้อมูลจากไฟล์ JSON มาแสดงผลเมื่อเปิดโปรแกรม"""
    if not os.path.exists(self.config_file):
      return
    try:
      with open(self.config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # โหลดค่า CSV Path
        if "csv_path" in data:
          self.csv_path.set(data["csv_path"])

        # โหลดรายการวัตถุดิบในตู้เย็น
        if "fridge_list" in data:
          self.fridge_list = data["fridge_list"]
          self.listbox.delete(0, tk.END)
          for item in self.fridge_list:
            self.listbox.insert(tk.END, item)

        # โหลดค่าสารอาหาร
        if "nutrients" in data:
          for label, val in data["nutrients"].items():
            if label in self.nutrient_entries:
              self.nutrient_entries[label].delete(0, tk.END)
              self.nutrient_entries[label].insert(0, val)
    except Exception as e:
      print(f"Error loading config: {e}")

  def on_closing(self):
    """ฟังก์ชันทำงานเมื่อผู้ใช้กดปิดหน้าต่าง"""
    self.save_data()
    self.root.destroy()

  # ==========================================  

  def run_recommendation(self):
    if not self.fridge_list:
      messagebox.showwarning(
          "แจ้งเตือน",
          "กรุณากรอกและกด Enter เพิ่มวัตถุดิบในตู้เย็นอย่างน้อย 1 อย่าง",
      )
      return

    try:
      target_nut = [
          float(self.nutrient_entries["Calories (kcal):"].get()),
          float(self.nutrient_entries["Protein (g):"].get()),
          float(self.nutrient_entries["Carbs (g):"].get()),
          float(self.nutrient_entries["Fat (g):"].get()),
      ]
    except ValueError:
      messagebox.showerror(
          "ข้อผิดพลาด", "กรุณากรอกค่าสารอาหารเป็นตัวเลขที่ถูกต้อง"
      )
      return

    csv_file = self.csv_path.get()
    try:
      recommendations = recommend_food_matrix(
          csv_file, self.fridge_list, target_nut
      )
    except FileNotFoundError:
      messagebox.showerror(
          "ข้อผิดพลาด", f"ไม่พบไฟล์ CSV ที่ระบุ: {csv_file}"
      )
      return
    except Exception as e:
      messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
      return

    # แสดงผลลัพธ์ใน Text Box
    self.result_text.delete("1.0", tk.END)
    if not recommendations:
      self.result_text.insert(
          tk.END, "ไม่พบเมนูอาหารที่ตรงกับวัตถุดิบในตู้เย็น (อย่างน้อย 10%)"
      )
      return
    top_3_recommendations = recommendations[:3]
    for idx, rec in enumerate(top_3_recommendations, 1):
      text_block = (
          f"อันดับที่ {idx}: {rec['food_name']}\n"
          f" - คะแนนรวม (Final Score): {rec['final_score']:.4f}\n"
          f"   (Cosine Sim: {rec['cosine_similarity']:.2f}, Euclidean Dist:"
          f" {rec['euclidean_distance']:.2f})\n"
          f" - วัตถุดิบทั้งหมด: {', '.join(rec['ingredients'])}\n"
          f" - สารอาหาร (Cal, Prot, Carb, Fat): {rec['nutrients']}\n"
          "--------------------------------------------------\n"
      )
      self.result_text.insert(tk.END, text_block)


if __name__ == "__main__":
  root = tk.Tk()
  app = FoodRecommenderApp(root)
  root.mainloop()