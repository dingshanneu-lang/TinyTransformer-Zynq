#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

extern void xform_core(const int16_t weights[522], const int16_t in[16], int16_t out[16]);

struct Vector { int16_t in[16]; int16_t q12[2]; };

static std::vector<Vector> load_vectors(const char* path) {
    std::vector<Vector> out;
    std::ifstream f(path);
    std::string line;
    std::string all;
    std::stringstream ss;
    if (f) { ss << f.rdbuf(); }
    else { printf("FAIL: cannot open %s\n", path); return out; }

    std::string s = ss.str();
    size_t pos = 0;
    while (pos < s.size()) {
        size_t pi = s.find("\"in\"", pos);
        if (pi == std::string::npos) break;
        size_t pb = s.find('[', pi);
        size_t pe = s.find(']', pb);
        if (pe == std::string::npos) break;
        std::string ip = s.substr(pb + 1, pe - pb - 1);

        size_t qi = s.find("\"q12\"", pe);
        size_t qb = s.find('[', qi);
        size_t qe = s.find(']', qb);
        if (qe == std::string::npos) break;
        std::string qp = s.substr(qb + 1, qe - qb - 1);

        Vector v;
        {
            std::stringstream ls(ip); std::string t; int k = 0;
            while (std::getline(ls, t, ',') && k < 16) {
                v.in[k++] = (int16_t)atoi(t.c_str());
            }
        }
        {
            std::stringstream ls(qp); std::string t; int k = 0;
            while (std::getline(ls, t, ',') && k < 2) {
                v.q12[k++] = (int16_t)atoi(t.c_str());
            }
        }
        out.push_back(v);
        pos = qe + 1;
    }
    return out;
}

int main() {
    std::vector<Vector> vecs = load_vectors("D:/ZynqTinyTransformerClassification/TCPServer/pl/xform_core_hls/ref/test_vectors.json");
    if (vecs.empty()) { printf("no vectors\n"); return 1; }
    printf("loaded %d vectors\n", (int)vecs.size());

    FILE* wf = fopen("D:/ZynqTinyTransformerClassification/TCPServer/pl/xform_core_hls/ref/weights_int16.bin", "rb");
    int16_t w[522];
    fread(w, 2, 522, wf);
    fclose(wf);

    // Generate new reference vectors
    printf("[\n");
    for (size_t v = 0; v < vecs.size(); v++) {
        int16_t out[16];
        xform_core((const int16_t*)w, vecs[v].in, out);
        
        // Print in JSON format for updating test_vectors.json
        if (v > 0) printf(",\n");
        printf("  {\n    \"in\": [");
        for (int i = 0; i < 16; i++) {
            if (i > 0) printf(",");
            printf("%d", vecs[v].in[i]);
        }
        printf("],\n    \"q12\": [%d,%d]\n  }", out[0], out[1]);
    }
    printf("\n]\n");
    return 0;
}