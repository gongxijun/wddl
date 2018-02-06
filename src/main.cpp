#include <iostream>

#include "../include/feature.h"
#include "../include/wddl.h"
int main() {

    wdl::Wddl *wddl = new wdl::Wddl();
    wddl->Run();
    MXNotifyShutdown();
    return EXIT_SUCCESS;
}