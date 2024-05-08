
AMAÇ

  Enerji sektörü, doğası gereği belirsizliklerle dolu bir yatırım alanıdır ve bu durum yatırımcılar için bir zorluk oluşturur. Bu bağlamda, zaman serisi analizleri ve makine öğrenmesi algoritmalarının kullanımı, yatırımcıların daha bilinçli kararlar vermelerine yardımcı olabilir. Zaman serisi analizleri, geçmiş fiyat hareketlerini ve ticaret hacimlerini inceleyerek gelecekteki eğilimleri tahmin etmeye odaklanır. Enerji sektörü hisse senetleri için bu tür analizler, piyasadaki dalgalanmaları anlamak ve gelecekteki fiyat hareketlerini öngörmek açısından kritiktir. Makine öğrenmesi algoritmaları ise büyük veri kümelerindeki desenleri belirleme konusunda üstündür. Enerji sektörü hisse senetleri için, bu algoritmalar fiyat tahminleri, risk analizi ve portföy yönetimi gibi alanlarda kullanılabilir. Regresyon analizi, karar ağaçları ve derin öğrenme modelleri gibi çeşitli algoritmalar, enerji hisselerindeki fiyat değişimlerini daha etkili bir şekilde tahmin etmeye yardımcı olabilir. Bu araştırma, Türkiye'deki enerji sektöründe önemli bir rol oynayan Enerjisa, Enka ve Yeo şirketlerinin hisse senetlerinin tahmin edilmesinde beş farklı makine öğrenimi modelini kullanacak ve zaman serileri tekniklerini uygulayacaktır. Elde edilen bulgular yorumlanacak ve analiz edilmiştir.

Zaman Serisi Analizi
Modellerin daha iyi sonuçlar üretebilmeleri için önceki bölümlerde de bahsedildiği üzere zaman serileri tekniklerinden ve yaklaşımlarından yararlanılacaktır. Geleneksel zaman serileri analizleri uygulanacak, mevsimsellik, döngü, trend ve durağanlık gibi bileşenler testler ile analiz edilecektir. SARIMA, ARIMA gibi modellerde yer alan değişkenler, makine öğrenimi modellerinin öğrenebilmesi ve daha performanslı çalışabilmelerini sağlamak adına üretilmiştir.

Zaman Yolu Grafiği



<img width="471" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/409d8177-9970-4f32-97ea-4dfab46d77dd">


Aşağıdaki tablolarda veri setinin durağan olup olmadığına yönelik yapılan ADF ve KPSS test sonuçları dikkate alındığında veri setinin durağan olmadığı tespit edilmiştir. 


ADF Sonucu

<img width="384" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/fcc5853d-d559-4343-bc9c-f62ddbeedb42">


KPSS  Sonucu

<img width="397" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/fc972dd0-61e8-4eea-b144-fcd35185cbab">


Çoklu Mevsimsel-Trend-Ayrışımı (MSTL)
Aşağıdaki grafikler incelendiğinde haftalık ve günlük mevsimsellik gözlenmekte, veri setine yeni değişkenler üretileceğinde bu mevsimsellik durumları gözetilerek üretilecektir.


Haftalık ve Günlük Mevsimellik

<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/b29e08fa-9676-4df9-8056-d1eb6f73e11c">


Çoklu Mevsimsellik Ayrışımı

<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/6b1e0227-ffa0-4287-9f86-593cffcefc46">

Değişkenlerin Üretilmesi
Veri setinde bir işlem yapmadan önce 2 adet değişken mevcuttu; tarih ve hisse fiyatı. Modellerin daha iyi öğrenebilmesi ve optimum tahmin sonuçları üretebilmeleri için zaman serilerindeki geleneksel modellerden olan SARIMA, ETS gibi modellerden esinlenerek gecikme değerleri, hareketli ortalama, ağırlıklı ortalama gibi değişkenler üretildi. Sonuç olarak toplamda 109 yeni değişken üretildi.


Eğitim Validasyon Seti


<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/43f9d599-73b2-4cb2-84f1-bafa3c81fe14">


XGBoost
Model belirtilen tarih aralığında eğitildi, sonrasında validasyon için geriye kalan tarih aralığının hisse senet fiyatları için tahminlemesi yapıldı. Parametre olarak n_estimators ve max_depth için ayarlamalar gerçekleştirildi ve n_estimators için 1000, max_depth için 3 değeri uygun görüldü. Tahminleme sonucunda model performansını değerlendirmek için SMAPE, MSE ve RMSE metrikleri kullanıldı. Aşağıda yer alan şekildeki SMAPE metriği sonucuna göre %25’lik bir hatayla hisse senetlerinin fiyat tahminlemesi gerçekleştirildi.

Gerçek Hisse Fiyatları ve Tahmin Sonuçları


<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/f208519c-bbbe-43e2-8dfc-814a43dd4759">


LightGBM
Xgboost modelinde gerçekleştirilen adımlar LightGBM eğitim ve validasyon setleri içinde uygulandı. Parametreler olarak; learning_rate 0.02, num_leaves 10, num_threads 4, max_depth 3 ve num_iterations 1000 olarak ayarlandı. SMAPE metriğine göre %22’lik bir hata ile tahminleme gerçekleştirildi.


 Gerçek Hisse Fiyatları ve Tahmin Sonuçları



<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/1f429adf-270c-401e-99e5-b805e20e3484">


Decision Tree
Decision Tree modeli içinde aynı eğitim ve validasyon işlemleri uygulandı. Parametreler olarak; max_depth 3, min_samples_split 2 seçildi ve ayarlandı. SMAPE metrik sonucuna göre %12’lik bir hata ile tahminlemeler gerçekleştirildi.

Gerçek Hisse Fiyatları ve Tahmin Sonuçları


<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/8732a2b4-16df-4575-9609-94441b22afa1">

SVM
Önceki modellere uygulanan işlemlerin aynısı uygulandı. Parametreler olarak kernel ‘rbf’, gamma 0.00001, C 1e3 ve epsilon 0.5 olarak ayarlandı. Tahminleme sonucunda SMAPE metriğine göre %4’lük bir hata ile tahminlemeler yapıldı. 


Test Seti Tahmin Sonucu


 
<img width="438" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/bbf9150a-907d-45e6-b071-427e087ec9a2">



Tahmin Trendi



<img width="411" alt="image" src="https://github.com/tolgaburak/Bachelor-s-thesis/assets/80509562/7298e0e2-97a5-4e14-b306-4376260bb5d6">




