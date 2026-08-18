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
değişen tek şey adresler ([Birden fazla makine](#4-birden-fazla-makine)).

---

## 1. Gereksinimler

| | |
|---|---|
| donanım | Jetson AGX Orin (JetPack 5.1 / L4T R35.x) |
| python | **3.8** — JetPack 5.1'in TensorRT bağlamaları sadece 3.8 için var |
| torch | NVIDIA'nın sistem paketi (CUDA'lı). `pip install torch` **yapma**, aarch64'te CUDA'sız sürüm gelir |
| ffmpeg | libx264 encoder'ıyla: `sudo apt install ffmpeg` |
| disk | kurulum sırasında geçici ~2.5 GB, kalıcı ~700 MB |

Kontrol:

```bash
python3.8 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -encoders | grep libx264
```

## 2. Kurulum — iki ayrı adım

Model kurulumu ile akış kurulumu **bilerek ayrı**. Biri ağırlık indirir, diğeri
ffmpeg/port/video hazırlar; birinin bozulması diğerini ilgilendirmez.

```bash
git clone https://github.com/defneyildirim00/lorat-b224-udp-orin.git
cd lorat-b224-udp-orin

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
   * LoRAT B-224 ağırlıkları (Google Drive klasörü; **~2 GB iner**, 52 MB'ı
     kullanılır, gerisi hemen silinir — tekil dosya linki yayınlanmamış, klasör
     de olduğu gibi inmek zorunda)
6. Duman testi: model kurulur, bir kare işlenir, süre yazdırılır

Tekrar çalıştırmak zararsız — var olanı atlar. Sadece doğrulamak için:
`./setup_model.sh --check` (hiçbir şey indirmez).

`setup_stream.sh` ne yapıyor: ffmpeg + libx264 var mı, UDP portları boş mu,
`net.core.rmem_max` yeterli mi, ve `videos/` boşsa test için hareketli hedefli
20 saniyelik bir klip üretir. Kendi videonu vermek için:
`./setup_stream.sh /yol/video.mp4`

## 3. Çalıştırma — üç terminal

**Hepsi tek makinede** — hiçbir adres yazmazsın, varsayılanlar `127.0.0.1`:

```bash
# TERMİNAL 1 — yayıncı (kamera rolü)
./publish.sh

# TERMİNAL 2 — tracker (kutuyu burada seçersin, kutular buraya yazılır)
./track.sh

# TERMİNAL 3 — izleyici
./view.sh
```

**İzleyici başka makinede** (Orin uçakta, operatör laptopta `192.168.1.50`):

```bash
# TERMİNAL 1 — ORIN'de. Tracker aynı makinede, adres gerekmez.
./publish.sh bike1.mp4

# TERMİNAL 2 — ORIN'de. İZLEYİCİNİN IP:PORT'U BURAYA YAZILIR.
VIEW=192.168.1.50:1235 ./track.sh

# TERMİNAL 3 — LAPTOPTA. Sadece kendi portu; IP yazılmaz.
./view.sh 1235
# repo laptopta kurulu değilse, aynı işi VLC de yapar:
vlc udp://@:1235
```

> İzleyicinin adresi terminal 3'te değil, **terminal 2'de** verilir — çünkü ona
> gönderen taraf tracker'dır. Gönderen karşıdakinin adresini yazar, alan sadece
> kendi portunu ([Portlar: iki atlama, dört ayar](#portlar-iki-atlama-dört-ayar)).

Sıra önemli: önce yayıncı, sonra tracker, en son izleyici. Tracker yayın
gelmeden açılırsa bekler, ölmez.

**Kapatırken** ters sırada: izleyici `q`, tracker `q`, yayıncı `Ctrl-C`.
Yayıncıyı unutursan port dolu kalır; `pkill -x ffmpeg`.

### Terminal 1 — hangi video yayınlanıyor

Video **sadece burada** seçilir. Tracker ile izleyici videoyu hiç bilmez; onlar
UDP'den ne gelirse onu alır (uçakta bu bacak zaten kameradır).

| komut | ne yapar |
|---|---|
| `./publish.sh` | `videos/` içindeki **ilk** mp4 (alfabetik) |
| `./publish.sh test_target.mp4` | `videos/` içinden isimle seç |
| `./publish.sh /yol/kendi_video.mp4` | klasör dışından, tam yol |
| `./publish.sh video.mp4 10.0.0.7:1234` | tracker başka makinedeyse onun adresi |

`videos/` içinde birden fazla mp4 varsa argümansız hal alfabetik ilkini alır ve bu
istediğin olmayabilir — ismi açıkça yaz. Klasörü `./setup_stream.sh` doldurur:
argümansız çalıştırırsan test klibi üretir, `./setup_stream.sh /yol/video.mp4`
dersen kendi videonu kopyalar (bu durumda test klibi üretilmez).

Yayın varsayılan olarak **döngüdedir**, klip bitince başa sarar:

```bash
LOOP=0 ./publish.sh              # bir kez oynat, bitince dur
FPS=25 ./publish.sh video.mp4    # kare hızını değiştir (tracker da aynı FPS'i kullanır)
```


### Terminal 2 — hedefi seçmek

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

### Terminal 3 — izleyici (view.sh ya da VLC)

`./view.sh` `udp://0.0.0.0:1235` dinler. **Dinleyen taraf olduğu için hiçbir IP'ye
ihtiyacı yoktur**; sadece portu bilmesi yeter, o da tracker'ın gönderdiği portla
aynı olmak zorunda.

```bash
./view.sh          # 1235 (varsayılan)
./view.sh 1236     # başka port — tracker da 1236'ya göndermeli
```

Portu iki tarafta birden değiştirmek gerekir, tek tarafta değiştirmek sessizce
siyah ekran demektir:

```bash
VIEW=127.0.0.1:1236 ./track.sh   # TERMİNAL 2: nereye gönderiyor
./view.sh 1236                   # TERMİNAL 3: nereyi dinliyor
```

Repo'nun kurulu olmadığı bir makinede izlemek için `view.sh` şart değil:
`vlc udp://@:1235` ya da `ffplay udp://0.0.0.0:1235` aynı işi görür.

### Portlar: iki atlama, dört ayar

Hat iki UDP atlamasından oluşur. Her atlamada **gönderen taraf karşıdakinin
adresini yazar; alan taraf sadece kendi portunu yazar.**

```
  TERMİNAL 1              TERMİNAL 2               TERMİNAL 3
  publish.sh              track.sh                 view.sh
  (gönderir)         (alır  ->  gönderir)          (alır)

      │  ATLAMA 1             │   ATLAMA 2             │
      └──── video ───────────►│──── kutulu görüntü ───►│
         kime: :1234       dinler: 1234            dinler: 1235
                           kime:   :1235
```

Terminal 2 ortada olduğu için **iki** ayar taşır: dinlediği port ve yolladığı
adres. Terminal 1 ile terminal 3'ün birer tane.

| atlama | gönderen — karşının adresini yazar | alan — kendi portunu yazar |
|---|---|---|
| 1 — video | terminal 1: `./publish.sh video.mp4 <TRACKER-IP>:1234` | terminal 2: `IN_PORT=1234 ./track.sh` |
| 2 — kutulu görüntü | terminal 2: `VIEW=<İZLEYİCİ-IP>:1235 ./track.sh` | terminal 3: `./view.sh 1235` |

Varsayılanlar `env.sh` içinde: `IN_PORT=1234`, `BIND=0.0.0.0`,
`VIEW=127.0.0.1:1235`. Tek seferlik değiştirmek için değişkeni komutun önüne yaz,
kalıcı istiyorsan `env.sh`'i düzenle.

**Hepsi tek makinedeyse hiçbirini yazmazsın** — varsayılanlar zaten `127.0.0.1`
gösteriyor, `./publish.sh` + `./track.sh` + `./view.sh` yeter. Adresler ancak
süreçler farklı makinelere dağılınca devreye girer
([Birden fazla makine](#4-birden-fazla-makine)).

Sık karışanlar:

* **"Terminal 1 zaten aynı makinede, neden orada port var?"** Çünkü gönderen o.
  UDP'de gönderen her zaman hedefin adresini yazar, aynı makinede bile — sadece o
  adres `127.0.0.1:1234` olduğu için varsayılan hallediyor ve sen yazmıyorsun.
* **"İzleyicinin portu terminal 2'de mi?"** Evet, `VIEW`'ün içinde. Tracker o
  porta yollar, izleyici o portu dinler; ikisi eşleşmezse izleyicide **siyah
  ekran** olur ve hiçbir hata mesajı çıkmaz.
* **"Terminal 2'de kaç port var?"** İki: `IN_PORT` (dinlediği) ve `VIEW`'ün portu
  (yolladığı). Aynı makinedeyken bunlar farklı olmak zorunda.
* **`VIEW=off ./track.sh`** — kutulu görüntüyü hiç yollamaz; terminal 3'e gerek
  kalmaz.

## 4. Birden fazla makine

Hangi ayarın nerede verildiği yukarıda:
[Portlar: iki atlama, dört ayar](#portlar-iki-atlama-dört-ayar). Burada sadece
makineler ayrıldığında değişenler var.

Kural değişmiyor — **gönderen tarafın karşıdakinin adresine ihtiyacı var, alan
tarafın hiçbir adrese ihtiyacı yok.** Yani tracker başka makinedeyse onun IP'sini
terminal 1'e, izleyici başka makinedeyse onun IP'sini terminal 2'ye yazarsın;
terminal 3'e hiçbir zaman IP yazılmaz, sadece port.

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

> **`--bbox` canlı yayında dikkatli kullan.** Yayına *ortadan* katılırsın: hangi
> karede başlayacağın belli değildir, dolayısıyla "0. karedeki" bir kutu genelde
> boş bir yere denk gelir ve tracker anında kaybeder. Sabit kutu ancak hedef
> büyük ve yavaşsa iş görür. Doğru yol pencereden seçmek; ekransız çalışacaksan
> kutuyu o anki görüntüye göre belirle.

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
lorat-b224-udp-orin/
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
