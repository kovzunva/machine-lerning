import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("reports/figures", exist_ok=True)

def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", 
                         edgecolor="none", facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize,
            ha="center", va="center", weight="bold")

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=2, mutation_scale=15))

# -------------------------------------------------------------
# Рисунок 1: Життєвий цикл переходу від ЛР №6 до ЛР №8
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

draw_box(ax, 0.4, 0.8, 1.8, 1.4, "ЛР №6\nhandoff/\n(manual_lr_2)", "#1e3a8a")
draw_arrow(ax, 2.2, 1.5, 2.8, 1.5)

draw_box(ax, 2.8, 0.8, 2.0, 1.4, "Схема даних &\nконфігурація\n(data_schema.json)", "#0284c7")
draw_arrow(ax, 4.8, 1.5, 5.4, 1.5)

draw_box(ax, 5.4, 0.8, 2.0, 1.4, "ML-компонент\nCLI & тести\n(src/ml_project)", "#059669")
draw_arrow(ax, 7.4, 1.5, 8.0, 1.5)

draw_box(ax, 8.0, 0.8, 1.6, 1.4, "ЛР №8\nREST API\n(FastAPI)", "#d97706")

plt.title("Перехід від експерименту до програмного ML-компонента", fontsize=12, pad=12, weight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig1_handoff_lifecycle.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# Рисунок 2: Схема валідації входу (SchemaValidator Flow)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

draw_box(ax, 0.3, 0.8, 1.5, 1.4, "Вхідний запит\n(JSON / CSV)", "#475569")
draw_arrow(ax, 1.8, 1.5, 2.4, 1.5)

draw_box(ax, 2.4, 0.8, 2.0, 1.4, "SchemaValidator\n(перевірка типів,\nNaN, Inf, діапазонів)", "#2563eb")
draw_arrow(ax, 4.4, 1.5, 5.0, 1.5)

draw_box(ax, 5.0, 0.8, 1.9, 1.4, "Впорядкування\nознак за схемою\n(30 ознак)", "#0d9488")
draw_arrow(ax, 6.9, 1.5, 7.5, 1.5)

draw_box(ax, 7.5, 0.8, 2.2, 1.4, "ProjectPredictor\npredict_proba()\n(версійована відповідь)", "#15803d")

plt.title("Послідовність перевірки inference-входу контрактом даних", fontsize=12, pad=12, weight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig2_schema_contract_flow.png", dpi=300)
plt.close()

print("[OK] Схеми fig1 та fig2 успішно згенеровано в reports/figures/")