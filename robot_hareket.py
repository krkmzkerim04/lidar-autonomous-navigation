"""
===========================================================
MOBİL ROBOTLAR DÖNEM PROJESİ

Konu:
LiDAR Tabanli Otonom Mobil Robot Navigasyonu

İçerik:
- Engel kaçinma
- Potansiyel alan yöntemi
- LiDAR simülasyonu
- LiDAR filtreleme
- Dead Reckoning lokalizasyonu
- Kalman Filter kestirimi
- RMSE hata analizi
- Gerçek zamanli animasyon

Hazirlayan:
Kerim Korkmaz

===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
from numpy.linalg import inv
from matplotlib.animation import FuncAnimation

# ===========================================================
# SİMÜLASYON DURUMU

simulation_finished = False

# ===========================================================
# HARİTA PARAMETRELERİ

map_width = 20
map_height = 20

# Engel listesi
# Format:
# (x, y, genişlik, yükseklik)

obstacles = [
    (2, 2, 2, 5),
    (6, 1, 2, 4),
    (10, 2, 3, 2),
    (15, 1, 2, 5),

    (3, 10, 2, 6),
    (7, 8, 3, 2),
    (12, 9, 2, 5),
    (16, 10, 2, 4),

    (5, 16, 4, 2),
    (11, 15, 3, 3)
]

# ===========================================================
# BAŞLANGIÇ VE HEDEF NOKTALARI

start = np.array([1.0, 1.0])
goal = np.array([18.0, 18.0])

# ===========================================================
# GERÇEK ROBOT DURUMU

x = start[0]
y = start[1]
theta = 0.0

# ===========================================================
# DEAD RECKONING DURUMU

dr_x = start[0]
dr_y = start[1]
dr_theta = 0.0

# ===========================================================
# EKF (GENİŞLETİLMİŞ KALMAN FİLTRESİ) MATRİSLERİ

# Durum Vektörü [x, y, theta]
kf_state = np.array([start[0], start[1], 0.0])

# Kovaryans Matrisi (P) - Başlangıçtaki belirsizlik
kf_P = np.eye(3)

# Süreç Gürültüsü (Q) - Odometri/IMU hatalarını temsil eder
kf_Q = np.diag([0.01, 0.01, 0.05]) 

# Ölçüm Gürültüsü (R) - LiDAR konumlandırma hatasını temsil eder
kf_R = np.diag([0.1, 0.1]) 

# Grafiklerde kullanmak için değişkenler
kf_x = kf_state[0]
kf_y = kf_state[1]
kf_theta = kf_state[2]

# ===========================================================
# YÖRÜNGE KAYITLARI

real_traj_x = []
real_traj_y = []

dr_traj_x = []
dr_traj_y = []

kf_traj_x = []
kf_traj_y = []

# ===========================================================
# HATA KAYITLARI

dr_errors = []
kf_errors = []

time_data = []

# ===========================================================
# SİMÜLASYON PARAMETRELERİ

dt = 0.1

lidar_range = 5.0
num_rays = 72

# ===========================================================
# GRAFİK PENCERESİ

fig, ax = plt.subplots(figsize=(8, 8))



# ===========================================================
# ENGEL KONTROL FONKSİYONU

# Bu fonksiyon verilen noktanın
# herhangi bir engelin içinde olup olmadığını kontrol eder.

def point_inside_obstacle(px, py):

    for obs in obstacles:

        ox, oy, w, h = obs

        if (ox <= px <= ox + w and
            oy <= py <= oy + h):

            return True

    return False

# ===========================================================
# A* (A-STAR) GLOBAL YOL PLANLAMA ALGORİTMASI

def heuristic(a, b):
    # İki nokta arasındaki Öklid mesafesi
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(node, grid_size):
    # 8 yönlü hareket (Sağ, Sol, Yukarı, Aşağı ve Çaprazlar)
    directions = [
        (0, grid_size), (0, -grid_size), (grid_size, 0), (-grid_size, 0),
        (grid_size, grid_size), (grid_size, -grid_size), (-grid_size, grid_size), (-grid_size, -grid_size)
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        # Harita sınırları dışına çıkmamak için kontrol
        if 0 <= nx <= map_width and 0 <= ny <= map_height:
            neighbors.append((nx, ny))
    return neighbors

def a_star_planning(start_pos, goal_pos, grid_size=0.5):
    # Başlangıç ve hedefi ızgaraya (grid) uydur
    start_node = (round(start_pos[0]/grid_size)*grid_size, round(start_pos[1]/grid_size)*grid_size)
    goal_node = (round(goal_pos[0]/grid_size)*grid_size, round(goal_pos[1]/grid_size)*grid_size)

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        # Hedefe çok yaklaştıysak yolu oluştur
        if heuristic(current, goal_node) < grid_size:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            path.append((goal_pos[0], goal_pos[1]))
            return path

        for neighbor in get_neighbors(current, grid_size):
            # Eğer bu komşu engelin içindeyse atla (0.2m güvenlik payı ile kontrol edilebilir)
            if point_inside_obstacle(neighbor[0], neighbor[1]):
                continue

            tentative_g = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [] # Yol bulunamazsa boş liste dön    

# ===========================================================
# BAŞLANGIÇTA A* İLE YOLU PLANLA

print("A* Algoritması rotayı planlıyor, lütfen bekleyin...")
planned_path = a_star_planning(start, goal, grid_size=0.5)

if planned_path:
    print("A* rotası başarıyla oluşturuldu!")
    planned_path_x = [p[0] for p in planned_path]
    planned_path_y = [p[1] for p in planned_path]
else:
    print("DİKKAT: A* hedefe giden bir yol bulamadı!")
    planned_path_x = []
    planned_path_y = []

# ===========================================================
# LiDAR SİMÜLASYONU

# Bu fonksiyon robot etrafında
# LiDAR taraması gerçekleştirir.

def simulate_lidar(robot_x, robot_y, robot_theta):

    lidar_points = []

    angles = np.linspace(
        -np.pi,
        np.pi,
        num_rays
    )

    for angle in angles:

        ray_angle = robot_theta + angle

        detected = False

        for r in np.linspace(0, lidar_range, 100):

            test_x = robot_x + r * np.cos(ray_angle)
            test_y = robot_y + r * np.sin(ray_angle)

            # Engel algılama kontrolü
            if point_inside_obstacle(test_x, test_y):

                # ===========================================================
                # LiDAR GÜRÜLTÜSÜ

                noise_x = np.random.normal(0, 0.03)
                noise_y = np.random.normal(0, 0.03)

                noisy_x = test_x + noise_x
                noisy_y = test_y + noise_y

                lidar_points.append((noisy_x, noisy_y))

                detected = True
                break

        # Engel algılanmazsa maksimum menzil kullanılır
        if not detected:

            end_x = robot_x + lidar_range * np.cos(ray_angle)
            end_y = robot_y + lidar_range * np.sin(ray_angle)

            lidar_points.append((end_x, end_y))

    return lidar_points

# ===========================================================
# LiDAR FİLTRELEME FONKSİYONU

# Bu fonksiyon LiDAR verisini
# Moving Average yöntemi ile filtreler.

def filter_lidar_points(lidar_points):

    filtered_points = []

    window_size = 3

    for i in range(len(lidar_points)):

        x_vals = []
        y_vals = []

        for j in range(
            max(0, i - window_size),
            min(len(lidar_points), i + window_size)
        ):

            x_vals.append(lidar_points[j][0])
            y_vals.append(lidar_points[j][1])

        filtered_x = np.mean(x_vals)
        filtered_y = np.mean(y_vals)

        filtered_points.append(
            (filtered_x, filtered_y)
        )

    return filtered_points

# ===========================================================
# LiDAR ENGEL KÜMELEME (CLUSTERING)

def cluster_lidar_points(lidar_points, threshold=0.8):
    if not lidar_points:
        return []
        
    clusters = []
    current_cluster = [lidar_points[0]]
    
    for i in range(1, len(lidar_points)):
        p1 = lidar_points[i-1]
        p2 = lidar_points[i]
        
        # İki nokta arasındaki Öklid mesafesi
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        if dist < threshold:
            current_cluster.append(p2)
        else:
            clusters.append(current_cluster)
            current_cluster = [p2]
            
    clusters.append(current_cluster)
    return clusters

# ===========================================================
# EKF FONKSİYONLARI

def ekf_predict(state, P, v, w, dt):
    x, y, theta = state[0], state[1], state[2]
    
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    new_theta = theta + w * dt
    predicted_state = np.array([new_x, new_y, new_theta])
    
    F = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0, 1]
    ])
    
    predicted_P = F @ P @ F.T + kf_Q
    
    return predicted_state, predicted_P

def ekf_update(state, P, z):
    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    y_res = z - (H @ state)
    S = H @ P @ H.T + kf_R
    K = P @ H.T @ inv(S)
    
    updated_state = state + K @ y_res
    updated_P = (np.eye(3) - K @ H) @ P
    
    return updated_state, updated_P

# ===========================================================
# ANA SİMÜLASYON DÖNGÜSÜ

def update(frame):

    global x, y, theta
    global dr_x, dr_y, dr_theta
    global kf_x, kf_y, kf_theta
    global simulation_finished

    # ===========================================================
    # HEDEF ÇEKİM KUVVETİ

    dx_goal = goal[0] - x
    dy_goal = goal[1] - y

    attractive_force_x = dx_goal
    attractive_force_y = dy_goal

    # ===========================================================
    # ENGEL İTME KUVVETİ

    repulsive_force_x = 0
    repulsive_force_y = 0

    lidar_points = simulate_lidar(x, y, theta)

    filtered_lidar = filter_lidar_points(
        lidar_points
    )

    # Filtrelenmiş noktaları kümele 
    lidar_clusters = cluster_lidar_points(filtered_lidar, threshold=0.8)
    
    for point in lidar_points:

        px, py = point

        dx_obs = x - px
        dy_obs = y - py

        distance = np.sqrt(
            dx_obs**2 +
            dy_obs**2
        )

        if distance < 2.0:

            if distance < 0.001:
                distance = 0.001

            repulsive_strength = 0.8 / (distance**2)

            repulsive_force_x += (
                repulsive_strength *
                dx_obs
            )

            repulsive_force_y += (
                repulsive_strength *
                dy_obs
            )

    # ===========================================================
    # TOPLAM KUVVET
    
    total_force_x = (
        attractive_force_x +
        repulsive_force_x
    )

    total_force_y = (
        attractive_force_y +
        repulsive_force_y
    )

    theta_goal = np.arctan2(
        total_force_y,
        total_force_x
    )

    angle_error = theta_goal - theta

    angle_error = np.arctan2(
        np.sin(angle_error),
        np.cos(angle_error)
    )

    # ===========================================================
    # KONTROL SİSTEMİ
    
    Kp = 2.5

    angular_velocity = Kp * angle_error

    linear_velocity = 0.3

    goal_distance = np.sqrt(
        dx_goal**2 +
        dy_goal**2
    )

    # ===========================================================
    # GERÇEK ROBOT HAREKETİ
    
    # ===========================================================
# HEDEFE ULAŞMA KONTROLÜ

    if goal_distance > 0.4:

        x += linear_velocity * np.cos(theta) * dt
        y += linear_velocity * np.sin(theta) * dt

        theta += angular_velocity * dt

    else:

        # Simülasyon sadece 1 kez sonlandırılsın
        if simulation_finished:
            return

        simulation_finished = True

        print("\nHedefe ulaşıldı.")

    # ===========================================================
    # SENSÖR GÜRÜLTÜLERİ

    encoder_noise_v = np.random.normal(0, 0.02)
    imu_noise_w = np.random.normal(0, 0.01)

    measured_v = linear_velocity + encoder_noise_v
    measured_w = angular_velocity + imu_noise_w

    # ===========================================================
    # DEAD RECKONING HESABI

    dr_x += measured_v * np.cos(dr_theta) * dt
    dr_y += measured_v * np.sin(dr_theta) * dt

    dr_theta += measured_w * dt

    # ===========================================================
    # GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF) UYGULAMASI
    
    global kf_state, kf_P
    
    # 1. Tahmin (Odometri ve IMU ile)
    kf_state, kf_P = ekf_predict(kf_state, kf_P, measured_v, measured_w, dt)
    
    # 2. Ölçüm Simülasyonu (LiDAR'dan gelen gürültülü konum verisi)
    lidar_noise_x = np.random.normal(0, 0.05)
    lidar_noise_y = np.random.normal(0, 0.05)
    z_measurement = np.array([x + lidar_noise_x, y + lidar_noise_y])
    
    # 3. Güncelleme (LiDAR Ölçümü ile EKF Düzeltmesi)
    kf_state, kf_P = ekf_update(kf_state, kf_P, z_measurement)
    
    # Çizim ve analiz için değişkenleri güncelle
    kf_x = kf_state[0]
    kf_y = kf_state[1]
    kf_theta = kf_state[2]

    # ===========================================================
    # YÖRÜNGE KAYDI
 
    real_traj_x.append(x)
    real_traj_y.append(y)

    dr_traj_x.append(dr_x)
    dr_traj_y.append(dr_y)

    kf_traj_x.append(kf_x)
    kf_traj_y.append(kf_y)

    # ===========================================================
    # HATA HESAPLARI

    dr_error = np.sqrt(
        (x - dr_x)**2 +
        (y - dr_y)**2
    )

    kf_error = np.sqrt(
        (x - kf_x)**2 +
        (y - kf_y)**2
    )

    dr_errors.append(dr_error)
    kf_errors.append(kf_error)

    time_data.append(frame * dt)

    # ===========================================================
    # GRAFİK ÇİZİMİ

    ax.clear()

    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)

    # Engellerin çizimi
    for obs in obstacles:

        ox, oy, w, h = obs

        rect = patches.Rectangle(
            (ox, oy),
            w,
            h,
            linewidth=1,
            edgecolor='black',
            facecolor='gray'
        )

        ax.add_patch(rect)

    # Başlangıç noktası
    ax.plot(
        start[0],
        start[1],
        'go',
        markersize=10,
        label='Başlangıç'
    )

    # Hedef noktası
    ax.plot(
        goal[0],
        goal[1],
        'ro',
        markersize=10,
        label='Hedef'
    )

    # Robot çizimi
    ax.plot(
        x,
        y,
        'bo',
        markersize=10,
        label='Robot'
    )

    # Robot yön oku
    ax.arrow(
        x,
        y,
        0.5 * np.cos(theta),
        0.5 * np.sin(theta),
        head_width=0.2,
        color='blue'
    )

    # ===========================================================
    # PLANLANAN YOL (A*) ÇİZİMİ
    
    if planned_path_x and planned_path_y:
        ax.plot(
            planned_path_x, 
            planned_path_y, 
            color='orange', 
            linestyle=':', 
            linewidth=2, 
            label='Planlanan Yol (A*)'
        )

    # Gerçek robot yolu
    ax.plot(
        real_traj_x,
        real_traj_y,
        'b-',
        linewidth=2,
        label='Gerçek Robot Yolu'
    )

    # Dead Reckoning yolu
    ax.plot(
        dr_traj_x,
        dr_traj_y,
        'm--',
        linewidth=2,
        label='Dead Reckoning Tahmini'
    )

    # Kalman Filter yolu
    ax.plot(
        kf_traj_x,
        kf_traj_y,
        'g-.',
        linewidth=2,
        label='Kalman Filter Tahmini'
    )

    # ===========================================================
    # HAM LiDAR VERİSİ

    for point in lidar_points:

        px, py = point

        ax.plot(
            px,
            py,
            'r.',
            markersize=2
        )

    # ===========================================================
    # FİLTRELENMİŞ LiDAR VERİSİ

    for point in filtered_lidar:

        px, py = point

        ax.plot(
            px,
            py,
            'c.',
            markersize=3
        )

        ax.plot(
            [x, px],
            [y, py],
            'g-',
            linewidth=0.3,
            alpha=0.2
        )

    # ===========================================================
    # GRAFİK AYARLARI

    ax.set_title(
        "LiDAR Tabanlı Otonom Mobil Robot Navigasyonu"
    )

    ax.set_xlabel("X Konumu (m)")
    ax.set_ylabel("Y Konumu (m)")

    ax.grid(True)

    ax.legend()

    ax.set_aspect('equal')

# ===========================================================
# ANİMASYON

ani = FuncAnimation(
    fig,
    update,
    frames=700,
    interval=40
)

plt.show()

# ===========================================================
# SON NAVİGASYON GRAFİĞİNİ KAYDET

fig.savefig(
    "final_navigation_result.png",
    dpi=300,
    bbox_inches='tight'
)

# ===========================================================
# RMSE HATA ANALİZİ

dr_rmse = np.sqrt(
    np.mean(np.array(dr_errors)**2)
)

kf_rmse = np.sqrt(
    np.mean(np.array(kf_errors)**2)
)

print("\n========== RMSE SONUÇLARI ==========")

print(
    f"Dead Reckoning RMSE Hatası : "
    f"{dr_rmse:.3f} m"
)

print(
    f"Kalman Filter RMSE Hatası  : "
    f"{kf_rmse:.3f} m"
)


# =========================
# RMSE HATA GRAFİĞİ

plt.clf()
plt.figure(figsize=(12, 6))

# Dead Reckoning Hatası
plt.plot(
    time_data,
    dr_errors,
    color='magenta',
    linewidth=2,
    label='Dead Reckoning Hatası'
)

# Kalman Filter Hatası
plt.plot(
    time_data,
    kf_errors,
    color='green',
    linewidth=2,
    label='Kalman Filter Hatası'
)

# Ortalama RMSE çizgileri
plt.axhline(
    dr_rmse,
    color='magenta',
    linestyle='--',
    linewidth=1.5,
    label=f'DR RMSE = {dr_rmse:.3f} m'
)

plt.axhline(
    kf_rmse,
    color='green',
    linestyle='--',
    linewidth=1.5,
    label=f'KF RMSE = {kf_rmse:.3f} m'
)

# Başlık
plt.title(
    "Konum Tahmin Hatalarının Zamana Göre Karşılaştırılması",
    fontsize=14
)

# Eksenler
plt.xlabel(
    "Zaman (s)",
    fontsize=12
)

plt.ylabel(
    "Konum Hatası (m)",
    fontsize=12
)

# Grid
plt.grid(True, linestyle='--', alpha=0.6)

# Açıklama Kutusu
plt.legend(fontsize=10)

# Görsel Kaydet
plt.savefig(
    "rmse_hata_analizi.png",
    dpi=300,
    bbox_inches='tight'
)
plt.show()