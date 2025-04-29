set -euo pipefail

echo "========== build start =========="

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

CUDA_ARCHITECTURE=80 
BUILD_TYPE=Release 
VERBOSE_MAKEFILE=OFF 
CODE_VERSION=Basic # -v: (Basic, ABFT, SNVR, Optimized)

while getopts ":v:" opt
do
    case $opt in
        v)
            CODE_VERSION=$OPTARG
            echo "CODE_VERSION: $CODE_VERSION"
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
        ?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done



echo_cmd() {
    echo $1
    $1
}


echo_cmd "rm -rf build output"
echo_cmd "mkdir build"

echo_cmd "cd build"
echo_cmd "cmake -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DFAI_VERBOSE_MAKEFILE=$VERBOSE_MAKEFILE -DCMAKE_INSTALL_PREFIX=$WORK_PATH/output -DCMAKE_SKIP_RPATH=ON .. -DCODE_VERSION=$CODE_VERSION"
echo_cmd "make -j"
echo_cmd "make install"

echo "========== build complete =========="
