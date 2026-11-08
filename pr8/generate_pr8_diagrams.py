import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("reports/figures", exist_ok=True)

def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=9.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", 
                         edgecolor="#334155", facecolor=color, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize,
            ha="center", va="center", weight="bold")

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=2, mutation_scale=14))

# -------------------------------------------------------------
# Рисунок 1: Архітектура навчального ML-компонента
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# Верхній ряд
draw_box(ax, 0.4, 3.0, 1.8, 1.2, "Клієнт", "#cbd5e1", text_color="#0f172a")
draw_arrow(ax, 2.2, 3.6, 2.9, 3.6)

draw_box(ax, 2.9, 3.0, 2.0, 1.2, "FastAPI\nконтракт", "#bbf7d0", text_color="#065f46")
draw_arrow(ax, 4.9, 3.6, 5.6, 3.6)

draw_box(ax, 5.6, 3.0, 2.0, 1.2, "Валідація\nсхеми", "#fef08a", text_color="#854d0e")
draw_arrow(ax, 7.6, 3.6, 8.3, 3.6)

draw_box(ax, 8.3, 3.0, 1.5, 1.2, "Predictor", "#e9d5ff", text_color="#581c87")

# Стрілка вниз від Predictor до Pipeline
draw_arrow(ax, 9.05, 3.0, 9.05, 2.0)

# Нижній ряд
draw_box(ax, 8.1, 0.8, 1.8, 1.2, "Pipeline +\nмодель", "#fecdd3", text_color="#9f1239")
draw_arrow(ax, 8.1, 1.4, 7.4, 1.4)

draw_box(ax, 5.4, 0.8, 2.0, 1.2, "Конфіг +\nметадані", "#fed7aa", text_color="#9a3412")
draw_arrow(ax, 5.4, 1.4, 4.7, 1.4)

draw_box(ax, 2.7, 0.8, 2.0, 1.2, "Логи й\nметрики", "#bae6fd", text_color="#075985")

# Стрілка вгору від Логів до FastAPI
draw_arrow(ax, 3.7, 2.0, 3.7, 3.0)

ax.text(5.0, 0.2, "Модель завантажується один раз; вхідні ознаки не записуються до технічних логів", 
        fontsize=9, ha="center", style="italic", color="#475569")

plt.title("Архітектура навчального ML-компонента", fontsize=12, pad=12, weight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig1_component_architecture.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# Рисунок 2: Життєвий цикл inference-запиту
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

draw_box(ax, 0.3, 1.0, 1.5, 1.2, "JSON-запит", "#dbeafe", text_color="#1e40af")
ax.text(1.05, 0.5, "request_id", fontsize=8.5, ha="center", color="#475569")
draw_arrow(ax, 1.8, 1.6, 2.3, 1.6)

draw_box(ax, 2.3, 1.0, 1.5, 1.2, "Контракт", "#dcfce7", text_color="#166534")
draw_arrow(ax, 3.8, 1.6, 4.3, 1.6)

draw_box(ax, 4.3, 1.0, 1.5, 1.2, "Валідація", "#fef9c3", text_color="#854d0e")
ax.text(5.05, 0.5, "schema_version + ranges", fontsize=8.5, ha="center", color="#475569")
draw_arrow(ax, 5.8, 1.6, 6.3, 1.6)

draw_box(ax, 6.3, 1.0, 1.5, 1.2, "Inference", "#f3e8ff", text_color="#6b21a8")
draw_arrow(ax, 7.8, 1.6, 8.3, 1.6)

draw_box(ax, 8.3, 1.0, 1.4, 1.2, "Відповідь", "#e0e7ff", text_color="#3730a3")
ax.text(9.0, 0.5, "model_version + warnings", fontsize=8.5, ha="center", color="#475569")

plt.title("Життєвий цикл inference-запиту", fontsize=12, pad=12, weight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig2_inference_lifecycle.png", dpi=300)
plt.close()

print("[OK] Рисунок 1 та Рисунок 2 успішно згенеровано у reports/figures/")