# Blackwell GPU Wiki

A standalone, educational reference for understanding NVIDIA's Blackwell generation — specifically the architectural split between **SM 10.0** (datacenter Blackwell) and **SM 12.0** (workstation/consumer Blackwell). Covers SM100 vs SM120 differences, NVFP4, `tcgen05`/TMEM, MoE inference, the kernel-library landscape, and compatibility patterns.

**Live site:** <https://blackwell-gpu-wiki.pages.dev/>

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000/>.

## Build

```bash
mkdocs build
# Output is in ./site
```

## Deployment

Hosted on Cloudflare Pages. Pushing to `main` triggers a rebuild.

Build settings in CF Pages:

- Build command: `pip install -r requirements.txt && mkdocs build`
- Build output directory: `site`
- Python version: 3.11+

## License

MIT. See [LICENSE](LICENSE).
