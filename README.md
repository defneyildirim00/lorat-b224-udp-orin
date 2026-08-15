# LoRAT B-224 — Jetson Orin üzerinde canlı UDP takip hattı

Tek nesne takibi (SOT) için **LoRAT B-224** modelini, üretimdeki gibi bir
ffmpeg → UDP → tracker → UDP hattında çalıştırır. Kutuyu fareyle seçersin,
kutulu görüntü tekrar UDP'ye basılır; istersen başka bir makineden izlersin.

```
  video/kamera                 ORIN                          izleyici
  ────────────       ─────────────────────────       ──────────────────
   ffmpeg -re   ──►  udp://:1234  ──►  tracker  ──►  udp://:1235  ──►  ffplay/VLC
   (publish.sh)      (track.sh: kutuyu sen seçersin, kutulu     (view.sh)
                      görüntüyü geri yayınlar)
```

Üçü de ayrı süreç ve **ayrı makinede olabilir**. Hepsi tek Orin'de de çalışır;
değişen tek şey adresler ([Birden fazla makine](#birden-fazla-makine)).

---

## 1. Gereksinimler

| | |
|---|---|
| donanım | Jetson AGX Orin (JetPack 5.1 / L4T R35.x) |
| python | **3.8** — JetPack 5.1'in TensorRT bağlamaları sadece 3.8 için var |
| torch | NVIDIA'nın sistem paketi (CUDA'lı). `pip install torch` **yapma**, aarch64'te CUDA'sız sürüm gelir |
| ffmpeg | libx264 encoder'ıyla: `sudo apt install ffmpeg` |
| disk | ~1.5 GB (üstkaynak kod + ağırlıklar) |

Kontrol:

```bash
python3.8 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -encoders | grep libx264
```

## 2. Kurulum — iki ayrı adım

Model kurulumu ile akış kurulumu **bilerek ayrı**. Biri ağırlık indirir, diğeri
ffmpeg/port/video hazırlar; birinin bozulması diğerini ilgilendirmez.

```bash
git clone https://github.com/defneyildirim00/lorat-b224-udp.git
cd lorat-b224-udp

./setup_model.sh      # ADIM 1: venv + bağımlılıklar + üstkaynak kod + ağırlıklar + duman testi
./setup_stream.sh     # ADIM 2: ffmpeg/port doğrulama + test videosu
```

`setup_model.sh` ne yapıyor:

1. `python3.8`, sistem torch'u ve CUDA'yı doğrular
2. `venv/` oluşturur — **`--system-site-packages` ile**, çünkü torch/torchvision/
   tensorrt NVIDIA'dan sistem paketi olarak gelir ve venv içine kurulamaz
3. `requirements.txt`'i kurar (timm, safetensors, gdown)
4. `third_party/` altına üstkaynak kodu **sabit commit'te** klonlar:
   * [LoRAT](https://github.com/LitingLin/LoRAT) (ECCV 2024) — `5260744`
5. Ağırlıkları indirir:
   * DINOv2 ViT-B/14 sırt kemiği (torch hub önbelleğine)
   * LoRAT B-224 ağırlıkları (Google Drive klasörü; ~360 MB iner, 52 MB'ı
     kullanılır, gerisi silinir — tekil dosya linki yayınlanmamış)
6. Duman testi: model kurulur, bir kare işlenir, süre yazdırılır

Tekrar çalıştırmak zararsız — var olanı atlar. Sadece doğrulamak için:
`./setup_model.sh --check` (hiçbir şey indirmez).

`setup_stream.sh` ne yapıyor: ffmpeg + libx264 var mı, UDP portları boş mu,
`net.core.rmem_max` yeterli mi, ve `videos/` boşsa test için hareketli hedefli
20 saniyelik bir klip üretir. Kendi videonu vermek için:
`./setup_stream.sh /yol/video.mp4`

## 3. Çalıştırma — üç terminal

```bash
# TERMİNAL 1 — yayıncı (kamera rolü)
./publish.sh

# TERMİNAL 2 — tracker (kutuyu burada seçersin, kutular buraya yazılır)
./track.sh

# TERMİNAL 3 — izleyici
./view.sh
```

Sıra önemli: önce yayıncı, sonra tracker, en son izleyici. Tracker yayın
gelmeden açılırsa bekler, ölmez.

**Kapatırken** ters sırada: izleyici `q`, tracker `q`, yayıncı `Ctrl-C`.
Yayıncıyı unutursan port dolu kalır; `pkill -x ffmpeg`.

### Hedefi seçmek

Model ~10-15 saniye yüklenir. **Yükleme bitene kadar bekle** — üstteki
`loading ...` yazısı kaybolup `no target - drag a box, or click its two corners`
çıkınca seç. Erken çizersen kutu saklanır ama 2 saniyeden eskiyse atılır (üstünde
çizildiği kare eskimiş olur, tracker garantili ıskalar).

| tuş | |
|---|---|
| sürükle | kutuyu çiz — takip başlar. Takip sırasında da çalışır, `r` gerekmez |
| tek tık + tek tık | köşe-köşe seçim. Sürükleme tutmuyorsa bunu kullan |
| `r` | hedefi bırak / yarım kalan seçimi iptal et |
| `f` | resmi dondur (küçük veya hızlı hedefe nişan alırken) |
| `q` / `ESC` | çık |

Fare tuşuna basılıyken resim durur, yani her zaman **duran bir görüntüye**
nişan alırsın ve kutu tam olarak o kareyle eşleşir.

### Terminale yazılanlar

```
[sel] box #1 sent to tracker (xywh) = [588, 220, 54, 131]
[sub] INIT #1 done, box (xywh) = [588, 220, 54, 131] — tracking from here
[box] #1      xywh=579,231,77,112  center=617,287  conf=0.57  step=24-26ms  lag=30-48ms
```

`sent` ile `INIT` sayıları ekranın altında da görünür (`sel: sent N / init M`) ve
uyuşmazsa kırmızı olur — kutunun tracker'a ulaşıp ulaşmadığını kesin ayırt eder.

Kutu yazımını seyreltmek: `./track.sh --print-box 10` (her 10. kare).

### İki gecikme, ve aynı şey değiller

* **step** — `track()` çağrısının o karede harcadığı süre. Modelin kendi maliyeti.
* **lag** — kutu hazır olduğunda karenin yaşı. Tracker eski kareyle meşgulken
  yeni karenin beklediği süreyi de içerir. **Operatörün gördüğü gecikme budur.**

Çıkışta ikisinin de medyan/p95'i, ısınma karesi ayrı satırda ve kaç karenin hiç
işlenemediği yazdırılır. İlk kare hariç tutulur: CUDA çekirdek seçimi ~1.5-2.5
saniye sürer ve bu ölçüm değildir — kurulumda sahte bir kutuyla önden ödenir, o
yüzden senin ilk kutun ilk kareden itibaren normal hızda takip edilir.

## 4. Birden fazla makine

Üç süreç, üç adres, her biri **farklı tarafta** ayarlanır:

| ne | nerede | kural |
|---|---|---|
| yayıncının hedefi | `./publish.sh video.mp4 <TRACKER-IP>:1234` | tracker'ın makinesi |
| tracker'ın dinlediği | `BIND=0.0.0.0` (varsayılan) | `127.0.0.1` **yazma** |
| tracker'ın hedefi | `VIEW=<İZLEYİCİ-IP>:1235 ./track.sh` | izleyecek makine |
| izleyicinin dinlediği | `./view.sh 1235` | zaten `0.0.0.0`, değişmez |

Kural: **gönderen tarafın karşıdakinin adresine ihtiyacı var, alan tarafın hiçbir
adrese ihtiyacı yok.**

`BIND=127.0.0.1` sadece loopback'i dinler; başka makineden gelen her paket sessizce
atılır — hata da vermez. Bu yüzden varsayılan `0.0.0.0`.

Uçaktaki gerçek dağılım (Orin'de ekran yok, operatör yerde):

```bash
# ORIN'de:
VIEW=192.168.1.50:1235 ./track.sh --headless --bbox 588,220,54,131

# yer istasyonunda (192.168.1.50):
./view.sh 1235          # repo oradaysa
vlc udp://@:1235        # repo yoksa — VLC her yerde çalışır
```

Ekransız modda kutu fareyle seçilemez, `--bbox x,y,w,h` ile verilir. Gerçek
serviste bu kutu operatörden ayrı bir UDP kontrol kanalıyla gelir; bu repo o
kanalı içermez.

Kendi IP'ni öğrenmek: `ip -4 addr` (Linux/Mac) ya da `ipconfig` (Windows).

## 5. Sorun giderme

| belirti | sebep |
|---|---|
| izleyicide siyah ekran | `--out`/`VIEW` adresi yanlış makineyi gösteriyor, ya da izleyicinin güvenlik duvarı UDP'yi kapatıyor. UDP'de bağlantı yok, hata da alamazsın |
| `Address already in use` | eski ffmpeg kalmış: `pkill -x ffmpeg` |
| kutu çizince hiçbir şey olmuyor | model daha yükleniyor olabilir; üstteki yazı `no target` olana kadar bekle. `--debug-mouse` her fare olayını yazar |
| `no publisher on :1234 yet` | yayıncı çalışmıyor ya da başka porta basıyor |
| tracker 30 fps'e yetişmiyor | normal — çıkıştaki "never reached the tracker" oranı kaç karenin atlandığını söyler |
| akış geliyor mu, kaba test | `ffmpeg -i udp://0.0.0.0:1235 -t 5 -f null -` |

## 6. Ölçümler (bu Orin'de, 1280×720 @30 fps)

| | |
|---|---|
| model kurulumu | 5-10 sn (sıcak önbellek) |
| CUDA ısınması | ~1.5 sn, kurulumda önden ödenir |
| kare başına (step) | **24-26 ms** |
| uçtan uca (lag) | **30-48 ms** |

LoRAT B-224 bu Orin'de 30 fps'lik bir beslemeye **yaklaşık yetişir**
(~40 FPS'lik model hızı). Yine de latest-frame-wins politikası gereği
işlenmeyen kareler olabilir; oran çıkışta yazar.

## 7. Dizin yapısı

```
lorat-b224-udp/
├── setup_model.sh      # ADIM 1: model
├── setup_stream.sh     # ADIM 2: akış
├── publish.sh          # terminal 1
├── track.sh            # terminal 2
├── view.sh             # terminal 3
├── env.sh              # ortak yollar/ayarlar (diğerleri source eder)
├── requirements.txt
├── install/smoke_test.py
├── src/                # inference + UDP kodu
│   ├── track_live.py     # UDP alıcı + pencere + kutu seçimi + geri yayın
│   ├── tracker_api.py    # tek model arayüzü (xywh)
│   ├── lorat_infer.py    # resmi LoRAT kodunu süren sarmalayıcı
│   ├── trt_lorat.py      # opsiyonel TensorRT motoru
│   └── view_udp.py       # izleyici
├── third_party/        # setup_model.sh klonlar (git'te değil)
├── weights/            # setup_model.sh indirir (git'te değil)
└── videos/             # setup_stream.sh üretir (git'te değil)
```

`third_party/`, `weights/`, `venv/`, `videos/` `.gitignore`'da: repo kod taşır,
gigabaytlarca ağırlık değil.

## 8. Kaynaklar ve lisans

* **LoRAT** — Lin et al., *Tracking Meets LoRA*, ECCV 2024.
  <https://github.com/LitingLin/LoRAT>
* **DINOv2** sırt kemiği — Meta AI.


Bu repodaki tutkal kod (UDP hattı, seçim arayüzü, kurulum) buraya ait; üstkaynak
projelerin kendi lisansları `third_party/` altında geçerlidir.
