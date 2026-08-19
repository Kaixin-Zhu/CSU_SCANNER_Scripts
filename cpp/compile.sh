g++ -c daphne.cpp -o daphne.o
ar rcs libdaphne.a daphne.o
#g++ plot_data.cpp daphne.cpp -o plot_data -I$ROOTSYS/include -L$ROOTSYS/lib -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lGui -L. -ldaphnei
g++ collect_data.cpp daphne.cpp -o collect_data \
    -std=c++17 \
    -L. -ldaphne \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lhdf5 -lhdf5_cpp \
    -lz \
    -Wl,-rpath,/opt/homebrew/lib \
    -Wl,-rpath,/opt/homebrew/include
  #-I$ROOTSYS/include -L$ROOTSYS/lib -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lGui -L. -ldaphne

echo "Building collect_data_periodic..."
g++ collect_data_periodic.cpp -o collect_data_periodic \
    -std=c++17 \
    -L. -ldaphne \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lhdf5 -lhdf5_cpp -lz \
    -Wl,-rpath,/opt/homebrew/lib


echo "Build finished successfully."
