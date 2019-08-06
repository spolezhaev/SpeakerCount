FROM conda/miniconda3

RUN apt-get update && apt-get install -y wget \
&& apt-get install git -y

ADD . /app
WORKDIR /app

RUN wget "https://downloader.disk.yandex.ru/disk/1aa5a40ceeb23906436d1ae1230b9b24977c3feb3acf38b372eb083b56857572/5d4956e2/qyC4_aHC109Fux4ZSQcRDDh9nyAJgI-MG_FSsPKaOvKEhwzkD1OBTmEe6DvCHCuAwhvmXuzgdjE0THEZ6XfzvA%3D%3D?uid=0&filename=processed_audio.tar.gz&disposition=attachment&hash=%2BK1pTzKe7PNd4i6flh/jbjFaDuaRQIHl0ZFVPwGgR8/SzKyqL0fgUctGBRxRYJrxq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Fx-gzip&owner_uid=141814092&fsize=2478676847&hid=cc8e77877542ffca887b9118c0c90153&media_type=compressed&tknv=v2" -O processed.tar.gz\
 && tar zxvpf processed.tar.gz && rm processed.tar.gz && mv processed data

RUN wget "https://downloader.disk.yandex.ru/disk/4155a59621257dda47d6743f153d778e756005266ede9dd2ac04b7e4f1307d58/5d4956ad/qyC4_aHC109Fux4ZSQcRDMcQZp-EsNtI6GCF7tX0dikKqzodVgeusRHNmP5q14rnw9Z6i3UPgAihLrGMsJPRtQ%3D%3D?uid=0&filename=saved_cv.tar.gz&disposition=attachment&hash=pXuFT/glkz0dmAQsZsEvHKtOFrlOS63JG9%2BYBFt4qB5w8M1Js6H6skxeXWfXP0ATq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Fx-gzip&owner_uid=141814092&fsize=6584242&hid=f33dbde96cc1eccea2fc19ebebb9bf85&media_type=compressed&tknv=v2" -O saved_cv.tar.gz \
&& tar zxvpf saved_cv.tar.gz && rm saved_cv.tar.gz

RUN pip install -r requirements.txt && conda install librosa -c conda-forge --yes