wget 'https://drive.google.com/uc?export=download&id=13vr06gv0MzrFmY3NoKLhi7tlHlv_q_pJ' -O train.txt
wget 'https://drive.google.com/uc?export=download&id=126G2yjVJdbgZlK18V_4QHdMv8Q7xZlpU' -O val.txt
wget 'https://drive.google.com/uc?export=download&id=1UwPx3xgdQqRqyURkqptWMwJsZqIt41d1' -O test.txt
wget 'https://drive.google.com/uc?export=download&id=1mUrzZnwRurvcd7XFQ9ky6nYt1P9gssef' -O ProcessImage.zip
unzip ProcessImage.zip
rm ProcessImage.zip

python train.py --config config.yaml
