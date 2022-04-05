#! /bin/bash
rsync -ave ssh --exclude '*.pyc' * ysloots@lilo6.science.ru.nl:/home/ysloots/masterproject/devimagine
