#! /bin/bash
rsync -ave ssh --exclude '*.pyc' * ysloots@astroluiz.science.ru.nl:/home/ysloots/masterproject/devimagine

