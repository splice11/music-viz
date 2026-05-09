"""GLSL shader sources for the offline music visualizer.

Pipeline:
  scene_fs   -> HDR float texture (the actual visual)
  bright_fs  -> extracts bright pixels for bloom
  blur_fs    -> separable gaussian, run H then V (3 mip levels)
  composite_fs -> tonemap + bloom add + chromatic aberration + grain + vignette
"""

# Shared fullscreen-triangle vertex shader.
FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID == 2) ? 3.0 : -1.0,
                  (gl_VertexID == 1) ? 3.0 : -1.0);
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Main scene shader.  All six section "modes" live here and are blended
# at section boundaries via u_modeBlend.
# ---------------------------------------------------------------------------
SCENE_FS = r"""
#version 330
in vec2 v_uv;
out vec4 frag;

uniform vec2  u_res;
uniform float u_time;
uniform int   u_modeA;
uniform int   u_modeB;
uniform float u_modeBlend;     // 0 = pure A, 1 = pure B
uniform float u_sectionT;      // 0..1 progress through current section

uniform float u_bass;          // bass stem rms, smoothed
uniform float u_bassPunch;     // bass stem onset env, sharp
uniform float u_drumsRms;
uniform float u_drumsPunch;    // smoothed onset envelope (kick/snare hits)
uniform float u_vocalsRms;
uniform float u_otherRms;
uniform float u_globalRms;
uniform float u_centroid;      // 0..1 normalized lead spectral centroid
uniform float u_pitch;         // 0..1 normalized lead pitch
uniform float u_beatPhase;     // 0..1 within beat
uniform float u_chroma[12];

// ---------- helpers ----------
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float hash3(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,37.719)))*43758.5453); }

vec2 rot(vec2 p, float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c)*p; }

float noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash(i), b = hash(i+vec2(1,0));
    float c = hash(i+vec2(0,1)), d = hash(i+vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

float fbm(vec2 p){
    float v = 0.0, a = 0.5;
    for (int i=0; i<5; i++){ v += a*noise(p); p = rot(p, 0.7)*2.03; a *= 0.5; }
    return v;
}

// Domain-warped FBM (cheap pseudo-fluid)
float warpedFbm(vec2 p, float t){
    vec2 q = vec2(fbm(p + vec2(0.0, t*0.15)), fbm(p + vec2(5.2, t*0.12)));
    vec2 r = vec2(fbm(p + 4.0*q + vec2(1.7, 9.2) + t*0.08),
                  fbm(p + 4.0*q + vec2(8.3, 2.8) + t*0.07));
    return fbm(p + 4.0*r);
}

// Convert HSV -> RGB
vec3 hsv(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz)*6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p-K.xxx, 0.0, 1.0), c.y);
}

// Dominant chroma -> hue (0..1).  Returns weighted circular mean over 12 pitch classes.
float chromaHue(){
    float sx = 0.0, sy = 0.0;
    for (int i=0; i<12; i++){
        float a = float(i) / 12.0 * 6.2831853;
        sx += u_chroma[i] * cos(a);
        sy += u_chroma[i] * sin(a);
    }
    float h = atan(sy, sx) / 6.2831853;
    return fract(h + 1.0);
}

// ----------------------------------------------------------------------
// MODE 0 — INTRO / AMBIENT BUILD
// Slow domain-warped clouds, soft drifting verticals, bass = pulse
// Palette: cyan / teal cryogenic
// ----------------------------------------------------------------------
vec3 mode_intro(vec2 uv){
    vec2 p = uv * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time * 0.08;
    float n = warpedFbm(p*1.6 + vec2(t, -t*0.3), u_time);
    n = pow(n, 1.4);

    // soft vertical light shafts
    float shaft = 0.0;
    for (int i=0; i<4; i++){
        float fi = float(i);
        float x = mod(p.x*0.5 + fi*0.31 + t*0.07, 1.0) - 0.5;
        shaft += exp(-pow(x*30.0,2.0)) * (0.4 + 0.6*sin(fi*1.7 + u_time*0.4));
    }
    shaft *= 0.25 * (0.4 + u_otherRms*1.4);

    float pulse = u_bassPunch * exp(-length(uv-0.5)*4.0);
    vec3 col = mix(vec3(0.02,0.08,0.14), vec3(0.35,0.85,1.0), n);
    col += vec3(0.4,0.7,1.0) * shaft;
    col += vec3(0.6,0.9,1.0) * pulse * 0.6;

    // breathing brightness with section progression
    col *= mix(0.6, 1.1, smoothstep(0.0, 0.9, u_sectionT));
    return col;
}

// ----------------------------------------------------------------------
// MODE 1 — VERSE / GROOVE
// Kaleidoscope mandala. Hue locked to chroma. Drum punch -> radial flash.
// ----------------------------------------------------------------------
vec3 mode_verse(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float ang = atan(p.y, p.x);
    float r   = length(p);

    // 8-fold kaleidoscope
    float seg = 6.2831853 / 8.0;
    ang = mod(ang, seg);
    ang = abs(ang - seg*0.5);
    vec2 kp = vec2(cos(ang), sin(ang)) * r;

    float t = u_time * 0.25;
    float n = fbm(kp*5.0 + vec2(t, -t*0.6) + vec2(u_drumsPunch*2.0));
    float rings = 0.5 + 0.5*sin(r*30.0 - u_time*2.0 - u_bass*4.0);
    float pat = mix(n, rings, 0.45);

    float hue = chromaHue();
    vec3 base = hsv(vec3(hue, 0.65, pat));
    base += vec3(1.0,0.5,0.9) * u_drumsPunch * exp(-r*3.0) * 0.8;
    base *= 0.7 + 0.6*u_globalRms;
    return base * 1.1;
}

// ----------------------------------------------------------------------
// MODE 2 — DROP / IMPACT
// Radial shockwave, exploding pseudo-particles, heavy contrast
// ----------------------------------------------------------------------
vec3 mode_drop(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float r = length(p);
    float ang = atan(p.y, p.x);

    // shockwave keyed to bass punch
    float wave = 0.0;
    for (int i=0; i<3; i++){
        float fi = float(i);
        float t = fract(u_time*0.4 + fi*0.33);
        float radius = t * 1.2;
        wave += exp(-pow((r - radius)*15.0, 2.0)) * (1.0 - t) * (0.6 + u_bass*1.5);
    }

    // pseudo-particles: hashed bright dots advected outward
    float parts = 0.0;
    for (int i=0; i<6; i++){
        float fi = float(i);
        float seed = hash(vec2(fi, 7.0));
        float a = seed * 6.2831853 + sin(u_time*0.3+fi)*0.4;
        float speed = 0.3 + seed*0.7;
        float life = fract(u_time*0.25 + seed);
        vec2 pos = vec2(cos(a), sin(a)) * life * 1.5;
        float d = length(p - pos);
        parts += exp(-d*40.0) * (1.0 - life) * (0.5 + u_drumsPunch);
    }

    // background spiral
    float spiral = 0.5 + 0.5*sin(ang*6.0 + r*15.0 - u_time*2.0);
    vec3 bg = mix(vec3(0.05,0.0,0.1), vec3(0.4,0.05,0.5), spiral);

    vec3 col = bg;
    col += vec3(1.0,0.4,0.9) * wave;
    col += vec3(1.0,0.95,1.0) * parts * 1.5;
    col += vec3(0.9,0.2,0.7) * u_bassPunch * 0.5;
    return col;
}

// ----------------------------------------------------------------------
// MODE 3 — MAIN / DRIVE
// Infinite tunnel.  Vocals draw a horizontal melody ribbon.
// ----------------------------------------------------------------------
vec3 mode_main(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float r = length(p);
    float ang = atan(p.y, p.x);

    // tunnel: depth = 1/r
    float z = 0.4 / max(r, 0.001) + u_time*0.6 + u_bass*0.3;
    float a = ang * 6.0 / 6.2831853;

    // grid pattern in tunnel UV
    vec2 tuv = vec2(a, z);
    float gx = abs(fract(tuv.x) - 0.5);
    float gy = abs(fract(tuv.y) - 0.5);
    float grid = smoothstep(0.45, 0.5, max(gx, gy));

    // chroma colored panels
    int panelIdx = int(mod(floor(tuv.y) + floor(tuv.x*8.0), 12.0));
    // GLSL doesn't allow dynamic index on array w/o ES310; unroll cheaply:
    float chr = 0.0;
    for (int i=0; i<12; i++) if (i == panelIdx) chr = u_chroma[i];

    float hue = chromaHue();
    vec3 panelCol = hsv(vec3(hue + 0.05*sin(tuv.x), 0.7, 0.4 + chr*0.6));
    vec3 col = panelCol * (1.0 - grid*0.6);

    // depth fog
    col *= exp(-r*0.4);

    // vocal melody ribbon — vertical position = pitch
    float ribbonY = (u_pitch - 0.5) * 0.8;
    float ribbon = exp(-pow((p.y - ribbonY)*40.0, 2.0)) * u_vocalsRms * 1.6;
    col += vec3(1.0, 0.85, 0.7) * ribbon;

    // beat-locked horizon flash
    float flash = pow(u_drumsPunch, 2.0) * exp(-abs(p.y)*8.0);
    col += vec3(1.0, 0.7, 0.4) * flash * 0.6;

    return col * (0.8 + 0.4*u_globalRms);
}

// ----------------------------------------------------------------------
// MODE 4 — BREAKDOWN / PULL-BACK
// Sparse dot grid, dim, tense; single bright cursor traces around.
// ----------------------------------------------------------------------
vec3 mode_breakdown(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);

    vec2 g = p * 18.0;
    vec2 gi = floor(g);
    vec2 gf = fract(g) - 0.5;
    float dot_ = exp(-dot(gf, gf)*30.0);
    float pulse = 0.4 + 0.6*sin(u_time*0.8 + (gi.x + gi.y)*0.5);
    float dim = 0.18 + 0.5*u_globalRms;
    vec3 col = vec3(0.4,0.6,0.8) * dot_ * pulse * dim;

    // single cursor that traces a lissajous, brightens on vocal hits
    vec2 cursor = vec2(sin(u_time*0.5), cos(u_time*0.31)) * 0.4;
    float cd = length(p - cursor);
    col += vec3(0.9,0.95,1.0) * exp(-cd*15.0) * (0.4 + u_vocalsRms*2.0);

    return col;
}

// ----------------------------------------------------------------------
// MODE 5 — OUTRO / RISE
// Vertical light beams growing from bottom; expanding rings.
// ----------------------------------------------------------------------
vec3 mode_outro(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time;

    // vertical beams
    float beams = 0.0;
    for (int i=0; i<7; i++){
        float fi = float(i);
        float x = (fi - 3.0) * 0.16;
        float h = 0.4 + 0.4*sin(fi*1.3 + t*0.5);
        h *= mix(0.3, 1.4, u_sectionT);
        float beam = exp(-pow((p.x - x)*40.0, 2.0));
        beam *= smoothstep(-0.5, h - 0.5, p.y);
        beams += beam * (0.5 + u_otherRms);
    }

    // rising rings
    float rings = 0.0;
    for (int i=0; i<4; i++){
        float fi = float(i);
        float r0 = fract(t*0.2 + fi*0.25);
        float radius = r0 * 1.3;
        float ring = exp(-pow((length(p) - radius)*12.0, 2.0));
        rings += ring * (1.0 - r0);
    }

    vec3 col = vec3(0.0);
    col += vec3(0.7,0.85,1.0) * beams * 0.9;
    col += vec3(1.0,0.9,0.7) * rings * 0.5;
    col += vec3(0.05,0.08,0.15);
    col *= 0.7 + 0.5*u_globalRms;
    return col;
}

vec3 dispatch(int mode, vec2 uv){
    if (mode == 0) return mode_intro(uv);
    if (mode == 1) return mode_verse(uv);
    if (mode == 2) return mode_drop(uv);
    if (mode == 3) return mode_main(uv);
    if (mode == 4) return mode_breakdown(uv);
    return mode_outro(uv);
}

void main(){
    vec3 a = dispatch(u_modeA, v_uv);
    vec3 b = dispatch(u_modeB, v_uv);
    vec3 col = mix(a, b, smoothstep(0.0, 1.0, u_modeBlend));

    // global beat-phase brightness lift
    col *= 1.0 + 0.06 * (1.0 - u_beatPhase);

    frag = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Bright extract for bloom.
# ---------------------------------------------------------------------------
BRIGHT_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex;
uniform float u_threshold;
void main(){
    vec3 c = texture(u_tex, v_uv).rgb;
    float b = max(max(c.r, c.g), c.b);
    float w = smoothstep(u_threshold, u_threshold + 0.4, b);
    frag = vec4(c * w, 1.0);
}
"""

# Separable gaussian blur (9-tap).
BLUR_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex;
uniform vec2 u_dir;     // pixel-space direction
uniform vec2 u_texel;
void main(){
    vec3 sum = vec3(0.0);
    float w[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    sum += texture(u_tex, v_uv).rgb * w[0];
    for (int i=1; i<5; i++){
        vec2 off = u_dir * u_texel * float(i);
        sum += texture(u_tex, v_uv + off).rgb * w[i];
        sum += texture(u_tex, v_uv - off).rgb * w[i];
    }
    frag = vec4(sum, 1.0);
}
"""

# Final composite: ACES tonemap + bloom add + chromatic aberration + grain + vignette.
COMPOSITE_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_time;
uniform float u_bloomAmt;
uniform float u_caAmt;
uniform float u_grainAmt;
uniform float u_vignette;
uniform float u_exposure;

vec3 aces(vec3 x){
    const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

float rand(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }

void main(){
    vec2 dir = v_uv - 0.5;
    float r2 = dot(dir, dir);

    // chromatic aberration: shift R/B along radial direction
    vec2 ofs = dir * u_caAmt;
    vec3 scene;
    scene.r = texture(u_scene, v_uv + ofs).r;
    scene.g = texture(u_scene, v_uv).g;
    scene.b = texture(u_scene, v_uv - ofs).b;

    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 col = scene + bloom * u_bloomAmt;
    col *= u_exposure;
    col = aces(col);

    // vignette
    col *= 1.0 - r2 * u_vignette;

    // film grain
    float g = rand(v_uv * vec2(1920.0, 1080.0) + u_time*60.0) - 0.5;
    col += g * u_grainAmt;

    // gamma to sRGB
    col = pow(max(col, 0.0), vec3(1.0/2.2));
    frag = vec4(col, 1.0);
}
"""
