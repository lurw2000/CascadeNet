PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# libpcap
wget https://www.tcpdump.org/release/libpcap-1.10.5.tar.xz
tar -xvf libpcap-1.10.5.tar.xz
cd libpcap-1.10.5
./configure --prefix=$PREFIX
make -j$(nproc)
make install


# nprint
wget https://github.com/nprint/nprint/releases/download/v1.2.1/nprint-1.2.1.tar.gz
tar -xvf nprint-1.2.1.tar.gz 
export CPPFLAGS="-I$PREFIX/include"
export LDFLAGS="-L$PREFIX/lib"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

cd nprint-1.2.1/
./configure --prefix=$PREFIX
make -j$(nproc)
make install

export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PREFIX/bin:$PATH
source ~/.bashrc
