int sum(int n) {
    int s = 0;
    int i = 0;
    while (i <= n) {
        s = s + i;
        i = i + 1;
    }
    return s;
}

int main() {
    int out = sum(10);
    return 0;
}
