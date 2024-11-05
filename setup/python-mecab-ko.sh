# sacrebleu mecab korean: manual install
# https://github.com/konlpy/konlpy/blob/master/scripts/mecab.sh
mkdir -p ~/tmp && cd ~/tmp
wget -O mecab-0.996-ko-0.9.2.tar.gz https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
sudo make install

# install mecab-ko-dic
mkdir -p ~/tmp && cd ~/tmp
wget -O mecab-ko-dic-2.1.1-20180720.tar.gz https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
# brew install autoconf automake
apt update && apt install -y autoconf automake
./autogen.sh
./configure
# mecab-config --libs-only-L | tee /etc/ld.so.conf.d/mecab.conf
# ldconfig
make
# sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
sudo make install
cat /usr/local/etc/mecabrc
# pip reinstall
pip install --upgrade --force-reinstall "mecab-ko" "mecab-ko-dic"
# checks
mecab --version
# mecab -d /usr/local/lib/mecab/dic/mecab-ko-dic
