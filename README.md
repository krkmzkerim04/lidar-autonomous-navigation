# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

Bu depo, 2B depo ortamında görev yapan bir otonom mobil robotun (AMR) temel navigasyon ve lokalizasyon algoritmalarını içeren Python tabanlı bir simülasyon projesini barındırmaktadır. 

Proje kapsamında robot; A* algoritması ile hedefe giden en kısa global rotayı bulmakta, LiDAR sensörü ile çevresini algılamakta, Yapay Potansiyel Alan (APF) yöntemi ile dinamik olarak engellerden kaçınmakta ve Genişletilmiş Kalman Filtresi (EKF) kullanarak gürültülü sensör verilerine rağmen kendi konumunu yüksek hassasiyetle tahmin etmektedir.

## 🚀 Proje Özellikleri

* **2B Depo Ortamı:** 20x20 metre boyutlarında, 10 adet statik engel içeren özel senaryo.
* **Global Planlama:** Grid tabanlı A* (A-Star) algoritması ile ideal rotanın çıkarılması.
* **Lokal Navigasyon:** Yapay Potansiyel Alan (APF) yöntemi ile reaktif engelden kaçınma.
* **Çevresel Algılama (LiDAR):** 360 derece ışın atımı (ray casting), Gaussian gürültü ekleme, Moving Average filtreleme ve Öklid tabanlı engel kümeleme.
* **Sensör Füzyonu ve Lokalizasyon:** Odometri ve LiDAR konum verilerinin Genişletilmiş Kalman Filtresi (EKF) ile birleştirilmesi.
* **Performans Analizi:** RMSE (Kök Ortalama Kare Hata) kullanılarak lokalizasyon performansının zaman serisi grafiklerinde incelenmesi.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

Simülasyon tamamen Python programlama dili kullanılarak sıfırdan geliştirilmiştir.
* `Python 3.x`: Ana programlama dili.
* `NumPy`: Matris hesaplamaları, EKF algoritmaları ve matematiksel işlemler.
* `Matplotlib`: 2B harita çizimleri, yörünge analizi ve gerçek zamanlı animasyon.

* ## ⚙️ Kurulum ve Çalıştırma Talimatları

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

**1. Depoyu Klonlayın:**
Sisteminize projeyi indirmek için terminal veya komut istemcisinde şu komutu çalıştırın:
```bash
git clone https://github.com/krkmzkerim04/lidar-autonomous-navigation.git
cd lidar-autonomous-navigation
python robot_hareket.py
