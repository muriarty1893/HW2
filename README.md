#  Logistic Regression ile Sınıflandırma Projesi

Bu proje, logistic regression kullanarak veri sınıflandırmayı amaçlayan bir makine öğrenimi çalışmasıdır. `logreg.py` dosyasında Logistic Regression sınıfı tanımlanmış ve sınıflandırma işlemi için gerekli metodlar (sigmoid, cost ve gradient hesaplama, eğitim ve tahmin yapma) Python’da uygulanmıştır. `test_logreg1.py` dosyası ise veri setini yükleyerek modelin eğitilmesini ve görselleştirilmesini sağlar.

## 📂 Dosya Yapısı

- **logreg.py**: LogisticRegression sınıfının ve ilgili fonksiyonların tanımlandığı ana dosya.
    - `__init__`: Hiperparametrelerin ve başlangıç değerlerinin belirlendiği yapıcı fonksiyon.
    - `sigmoid`: Sigmoid fonksiyonunun tanımlandığı metot, modelin doğrusal kombinasyonlarını olasılığa dönüştürür.
    - `computeCost`: Düzenlemeli (regularized) cost fonksiyonunu hesaplar, modelin tahmin doğruluğunu ölçer.
    - `computeGradient`: Düzenlemeli gradient hesaplaması yaparak cost fonksiyonuna göre ağırlıkları günceller.
    - `fit`: Gradyan iniş (gradient descent) kullanarak modeli eğitir ve en uygun ağırlıkları bulur.
    - `predict`: Eğitilen model kullanılarak yeni veri için tahmin yapılmasını sağlar.
  
- **test_logreg1.py**: Modelin `logreg.py` dosyasını kullanarak eğitilmesini ve performansının değerlendirilmesini sağlayan test dosyası. Eğitim verisi üzerinde logistic regression modelini eğitir ve görselleştirir.

## 📊 Veri Seti
Test scripti içinde kullanılan veri seti `data1.dat` adlı dosyadan yüklenmektedir. Bu veri seti, iki farklı özellik ve ikili sınıflandırma hedefi içerir. Eğitim verisi, logistic regression modelini eğitmek ve karar sınırını çizmek için kullanılır.

## 🛠 Kullanılan Kütüphaneler
- **Numpy**: Veriler üzerinde matematiksel işlemler gerçekleştirmek için.
- **Matplotlib**: Eğitim verilerini ve modelin karar sınırını görselleştirmek için.

## 🔍 Çalıştırma Talimatları
1. **Python Ortamını Kurun**: Proje Python 3.x ile çalışmaktadır. Gerekli kütüphaneleri yüklemek için:
   ```bash
   pip install numpy matplotlib
   ```
2. **Veri Setini Hazırlayın**: `data1.dat` dosyasını aynı dizine ekleyin.
3. **Modeli Eğit ve Test Et**: `test_logreg1.py` dosyasını çalıştırarak logistic regression modelini eğitebilir ve karar sınırını görselleştirebilirsiniz:
   ```bash
   python test_logreg1.py
   ```
