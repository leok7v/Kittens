// Tiny CLI wrapper around CEPhonemizer's IPAPhonemizer.
// Usage: phonemize-cli <rules_path> <list_path> "<text>"
// Prints IPA to stdout, one line per text arg.
#include "phonemizer.h"
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <rules_path> <list_path> <text>\n";
        return 1;
    }
    IPAPhonemizer p(argv[1], argv[2], "en-us");
    if (!p.isLoaded()) {
        std::cerr << "load failed: " << p.getError() << "\n";
        return 1;
    }
    // All remaining args become a single space-joined text.
    std::string text = argv[3];
    for (int i = 4; i < argc; i++) { text += " "; text += argv[i]; }
    std::cout << p.phonemizeText(text) << std::endl;
    return 0;
}
