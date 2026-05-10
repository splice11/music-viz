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

// =====================================================================
// HELPERS
// =====================================================================

const float PI  = 3.14159265359;
const float TAU = 6.28318530718;

mat2 rot2(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

// Hash without sine (David Hoskins). No diagonal banding artefacts.
float hash11(float p){
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}
float hash12(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3){
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 hash33(vec3 p3){
    p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float vnoise2(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash12(i), b = hash12(i+vec2(1,0));
    float c = hash12(i+vec2(0,1)), d = hash12(i+vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

float vnoise3(vec3 p){
    vec3 i = floor(p), f = fract(p);
    vec3 u = f*f*(3.0-2.0*f);
    float n000 = hash13(i+vec3(0,0,0));
    float n100 = hash13(i+vec3(1,0,0));
    float n010 = hash13(i+vec3(0,1,0));
    float n110 = hash13(i+vec3(1,1,0));
    float n001 = hash13(i+vec3(0,0,1));
    float n101 = hash13(i+vec3(1,0,1));
    float n011 = hash13(i+vec3(0,1,1));
    float n111 = hash13(i+vec3(1,1,1));
    return mix(
        mix(mix(n000,n100,u.x), mix(n010,n110,u.x), u.y),
        mix(mix(n001,n101,u.x), mix(n011,n111,u.x), u.y),
        u.z);
}

float fbm2(vec2 p){
    float v = 0.0, a = 0.5;
    for (int i=0; i<5; i++){ v += a*vnoise2(p); p = rot2(0.7)*p*2.03; a *= 0.5; }
    return v;
}

float fbm3(vec3 p){
    float v = 0.0, a = 0.5;
    for (int i=0; i<5; i++){ v += a*vnoise3(p); p = p*2.02 + 17.0; a *= 0.5; }
    return v;
}

// Ridge noise: sharper, mountain-like.
float ridge2(vec2 p, int oct){
    float v = 0.0, a = 0.5, freq = 1.0;
    for (int i=0; i<6; i++){
        if (i >= oct) break;
        float n = vnoise2(p*freq);
        n = 1.0 - abs(n*2.0 - 1.0);
        v += a * n*n;
        freq *= 2.07; a *= 0.5;
    }
    return v;
}

vec3 hsv(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz)*6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p-K.xxx, 0.0, 1.0), c.y);
}

float chromaHue(){
    float sx = 0.0, sy = 0.0;
    for (int i=0; i<12; i++){
        float a = float(i) / 12.0 * TAU;
        sx += u_chroma[i] * cos(a);
        sy += u_chroma[i] * sin(a);
    }
    return fract(atan(sy, sx) / TAU + 1.0);
}

float chromaAt(int idx){
    float v = 0.0;
    for (int i=0; i<12; i++) if (i == idx) v = u_chroma[i];
    return v;
}

// =====================================================================
// MODE 0 — VOLUMETRIC WATER / ICE  (intro)
// Top-down raymarched height-field surface with caustics underneath.
// Bass produces a real radial swell; drum hits ripple outward.
// =====================================================================
float waterHeight(vec2 p, float t){
    // base ocean
    float h = 0.0;
    vec2 q = p;
    h += 0.55 * vnoise2(q*1.4 + vec2(t*0.18, t*0.07));
    h += 0.30 * vnoise2(q*3.1 + vec2(-t*0.12, t*0.21));
    h += 0.18 * vnoise2(q*6.7 + vec2(t*0.31, -t*0.17));

    // bass swell — concentric standing wave breathing from center
    float r = length(p);
    float swell = sin(r*4.0 - t*1.6) * exp(-r*0.6);
    h += swell * (0.05 + u_bass*0.18);

    // beat-locked expanding ring (sharp)
    float ringR = u_beatPhase * 1.6;
    float ring = exp(-pow((r - ringR)*8.0, 2.0));
    h += ring * (0.02 + u_drumsPunch*0.15);

    return h;
}

vec3 waterNormal(vec2 p, float t){
    float e = 0.005;
    float hx = waterHeight(p + vec2(e,0), t) - waterHeight(p - vec2(e,0), t);
    float hy = waterHeight(p + vec2(0,e), t) - waterHeight(p - vec2(0,e), t);
    return normalize(vec3(-hx, 2.0*e, -hy));
}

vec3 mode_water(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0) * 2.4;
    float t = u_time;

    vec3 n = waterNormal(p, t);
    // tilted view so the surface is seen at an angle — gives proper fresnel
    // and makes the water read as a reflective surface, not a flat texture.
    vec3 viewDir = normalize(vec3(0.0, 0.55, 0.84));
    vec3 sunDir  = normalize(vec3(0.6, 0.8, 0.3));

    // fresnel for shallow grazing — boosted so flat water still reflects sky
    float fres = pow(1.0 - max(dot(n, viewDir), 0.0), 2.5);
    fres = clamp(fres + 0.15, 0.0, 1.0);

    // sky color (what reflects)
    vec3 sky = mix(vec3(0.02, 0.12, 0.22), vec3(0.55, 0.85, 1.05), 0.5 + 0.5*n.y);

    // refracted sub-surface — caustic field underneath
    vec2 caustP = p + n.xz*0.4;
    float c1 = vnoise2(caustP*4.0 + vec2(t*0.4, -t*0.3));
    float c2 = vnoise2(caustP*7.0 - vec2(t*0.27, t*0.41));
    float caustic = pow(max(c1*c2, 0.0), 5.0);
    vec3 sub = mix(vec3(0.005, 0.03, 0.08), vec3(0.06, 0.32, 0.55), 0.5 + 0.5*caustic);
    sub += vec3(0.5, 0.85, 1.0) * caustic * 0.18;

    // specular sun glint — sharp & beat-modulated
    vec3 H = normalize(viewDir + sunDir);
    float spec = pow(max(dot(n, H), 0.0), 90.0) * 8.0;
    spec *= (0.6 + u_drumsPunch*1.6);

    vec3 col = mix(sub, sky, fres);
    col += vec3(1.0, 0.97, 0.9) * spec;

    // deep-water vignetting toward edges adds a sense of depth
    col *= mix(0.55, 1.15, smoothstep(1.6, 0.2, length(p)));

    // bass adds a cool breath of light in the centre
    col += vec3(0.15, 0.4, 0.7) * u_bass * 0.35 * exp(-length(p)*1.4);

    return col;
}

// =====================================================================
// MODE 1 — JOY DIVISION RIDGES  (verse / groove)
// Stacked silhouette mountain ridges from back to front.
// Each ridge is displaced by music; front ridges occlude back ones.
// =====================================================================
float ridgeHeight(float x, float idx, float t){
    // per-ridge seed so the band doesn't move in lockstep
    float seed = idx * 7.13;
    // sharp ridge layer — gives mountain-like spikes (NOT smooth wave)
    float h  = 0.70 * ridge2(vec2(x*1.2 + seed, idx*0.31 + t*0.04), 5);
    // low-frequency big shape, modulated by bass — drives a slow swell
    h += 0.55 * (0.3 + u_bass) * vnoise2(vec2(x*0.5 + seed*0.5, idx*0.11));
    // a strong central "main peak" that breathes with bass punch
    float center = exp(-pow(x*0.9, 2.0));
    h += center * (0.3 + u_bassPunch * 0.9 + u_vocalsRms * 0.5) * (1.0 - idx*0.7);
    // a couple of secondary peaks at semi-random positions per ridge
    float peakX = sin(seed*3.7) * 1.6;
    float secondary = exp(-pow((x - peakX)*1.3, 2.0));
    h += secondary * (0.15 + u_otherRms*0.35) * (1.0 - idx*0.6);
    // high-freq jitter from drums (texture, not bulk)
    h += 0.12 * u_drumsRms * vnoise2(vec2(x*9.0 + seed, t*0.6));
    return h;
}

vec3 mode_ridges(vec2 uv){
    float aspect = u_res.x / u_res.y;
    // screen-space coords: x in [-aspect, aspect], y in [0,1] bottom-to-top
    vec2 p = vec2((uv.x - 0.5) * 2.0 * aspect, uv.y);

    // background — deep night with subtle scan field
    vec3 bg = mix(vec3(0.005, 0.008, 0.018), vec3(0.02, 0.025, 0.05),
                  smoothstep(0.0, 1.0, p.y));
    // tiny stars in upper half
    if (p.y > 0.45){
        vec2 sp = floor(p * 220.0);
        float s = step(0.997, hash12(sp)) * smoothstep(0.45, 1.0, p.y);
        bg += vec3(0.7, 0.8, 1.0) * s * 0.6;
    }

    const int N_RIDGES = 36;
    float topY = 0.82;       // back-most ridge baseline
    float botY = 0.05;       // front-most ridge baseline
    float spacing = (topY - botY) / float(N_RIDGES - 1);
    float t = u_time;

    vec3 col = bg;
    bool covered = false;

    // walk front-to-back; first ridge whose silhouette covers this pixel wins
    for (int i = 0; i < N_RIDGES; i++){
        float idx = float(i) / float(N_RIDGES - 1);   // 0=front, 1=back
        float baseY = botY + idx * (topY - botY);

        // perspective: front ridges are dramatically taller on screen
        float amp = mix(0.32, 0.05, idx);
        float h = ridgeHeight(p.x * mix(1.0, 0.5, idx), idx, t);
        float yCurve = baseY + h * amp;

        if (p.y <= yCurve){
            // distance to the silhouette line, in pixels-ish
            float d = (yCurve - p.y);
            // stroke gets thinner for back ridges → real depth cue
            float strokeW = mix(0.0050, 0.0018, idx);
            float strokeI = smoothstep(strokeW*1.6, 0.0, d);

            // fill — darker than bg so ridges read as solid
            float depthFade = mix(1.0, 0.35, idx);   // back ridges dimmer
            vec3 fill = vec3(0.008, 0.011, 0.022) * depthFade;

            // the stroke colour borrows hue from chroma so it dances
            float hue = chromaHue();
            vec3 stroke = mix(vec3(0.95, 0.97, 1.0),
                              hsv(vec3(hue, 0.55, 1.0)),
                              0.30 + 0.45*u_otherRms);
            stroke *= depthFade * (0.7 + 0.6*u_globalRms);

            col = mix(fill, stroke, strokeI);

            // soft glow under the line for the front-most few ridges
            if (idx < 0.25){
                float glow = exp(-d*45.0) * (1.0 - idx*4.0);
                col += stroke * glow * 0.5 * (0.4 + u_drumsPunch);
            }

            covered = true;
            break;
        }
    }

    // sky tint above all ridges gets a soft pulse with bass
    if (!covered){
        col += vec3(0.05, 0.08, 0.18) * u_bass * 0.4 * smoothstep(0.4, 1.0, p.y);
    }

    return col;
}

// =====================================================================
// MODE 2 — PLASMA CORE  (drop / impact)
// Raymarched glowing displaced sphere + sharp expanding shockwaves.
// Bass punch deforms the surface; drum punch fires shockwaves.
// =====================================================================
float coreSDF(vec3 p, float t){
    float r = 0.55 + 0.08*u_bass + 0.06*sin(t*2.0);
    // surface displacement — turbulence with punch spike
    float disp = fbm3(p*2.6 + vec3(t*0.4, -t*0.3, t*0.27));
    disp += u_bassPunch * 0.35 * vnoise3(p*5.0 + t);
    return length(p) - r - disp*0.18;
}

vec3 mode_plasma(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time;

    vec3 ro = vec3(0.0, 0.0, -2.4);
    vec3 rd = normalize(vec3(p, 1.4));

    // slow camera roll on beatPhase so the geometry feels alive
    float roll = u_beatPhase * 0.25;
    rd.xy = rot2(roll) * rd.xy;

    // raymarch
    float td = 0.0;
    float hit = -1.0;
    vec3 hitP = vec3(0.0);
    for (int i = 0; i < 64; i++){
        vec3 q = ro + rd * td;
        float d = coreSDF(q, t);
        if (d < 0.002){ hit = td; hitP = q; break; }
        if (td > 6.0) break;
        td += max(d*0.85, 0.005);
    }

    // background: dark cosmos with a magenta dust glow
    vec2 sp = p * 1.4;
    float bgF = fbm2(sp*2.0 + vec2(t*0.05, t*0.03));
    vec3 bg = mix(vec3(0.02, 0.0, 0.04), vec3(0.18, 0.02, 0.18), pow(bgF, 2.0));

    // expanding sharp shockwave rings keyed to the drum punch
    float rings = 0.0;
    for (int i = 0; i < 4; i++){
        float fi = float(i);
        float life = fract(t*0.55 + fi*0.27);
        float rad = life * 1.6;
        float band = exp(-pow((length(p) - rad)*55.0, 2.0));
        rings += band * (1.0 - life) * (0.3 + u_drumsPunch*1.4);
    }
    bg += vec3(1.0, 0.35, 0.7) * rings;

    if (hit < 0.0){
        return bg;
    }

    // shading on the displaced sphere
    float e = 0.01;
    vec3 nrm = normalize(vec3(
        coreSDF(hitP + vec3(e,0,0), t) - coreSDF(hitP - vec3(e,0,0), t),
        coreSDF(hitP + vec3(0,e,0), t) - coreSDF(hitP - vec3(0,e,0), t),
        coreSDF(hitP + vec3(0,0,e), t) - coreSDF(hitP - vec3(0,0,e), t)
    ));

    vec3 lightDir = normalize(vec3(0.6, 0.5, -1.0));
    float diff = max(dot(nrm, lightDir), 0.0);
    float fres = pow(1.0 - max(dot(nrm, -rd), 0.0), 3.0);

    // emission from the surface — kept in the warm red/orange/magenta range
    // (centroid only nudges within that band rather than driving full hue).
    float hue = fract(0.97 + 0.06 * u_centroid);   // ~0.97..0.03 (red→orange)
    vec3 emission = hsv(vec3(hue, 0.85, 0.9));

    // surface temperature varies with the local fbm at the hit
    float temp = fbm3(hitP*4.0 + t*0.6);
    emission *= 0.5 + 1.0*temp;

    // bright rim from fresnel + diffuse from a cool back-light
    vec3 col = emission * (0.4 + 0.5*diff);
    col += vec3(1.0, 0.45, 0.55) * fres * (0.5 + u_bassPunch*1.0);

    // bass punch flashes the whole core (gentle)
    col *= 1.0 + u_bassPunch * 0.6;

    // composite over background through a soft fall-off
    return col + bg * 0.3;
}

// =====================================================================
// CLOUD VOLUME — used by MODE 3 (clouds+lightning) and MODE 5 (aurora).
// Ray marches a horizontally-bounded cloud slab.
// =====================================================================
// Cloud slab spans y in [SLAB_LO, SLAB_HI]. Camera sits below it.
const float SLAB_LO = 0.6;
const float SLAB_HI = 1.6;

float cloudDensity(vec3 p, float t){
    // outside the slab, density is 0 (and we should be skipping these samples)
    if (p.y < SLAB_LO || p.y > SLAB_HI) return 0.0;

    // base shape — large soft puffs
    vec3 q = p * 0.7 + vec3(t*0.04, 0.0, t*0.02);
    float base = fbm3(q);
    // erode with hi-freq detail to make the cloud edges wispy and create gaps
    float detail = fbm3(p*2.8 + vec3(-t*0.08, t*0.03, t*0.05));
    float d = base * 1.4 - 0.78 - 0.22*(1.0 - detail);

    // vertical falloff inside the slab — denser in the middle
    float h = (p.y - SLAB_LO) / (SLAB_HI - SLAB_LO);
    float vf = smoothstep(0.0, 0.25, h) * smoothstep(1.0, 0.6, h);

    return max(d, 0.0) * vf;
}

// Triggered lightning bolt: deterministic per ~half second, gated by drum hits.
vec3 lightningPos(float t){
    float seg = floor(t * 2.0);
    return vec3((hash11(seg*1.7) - 0.5) * 5.0,
                (SLAB_LO + SLAB_HI)*0.5 + (hash11(seg*3.1) - 0.5) * 0.3,
                2.5 + hash11(seg*5.2) * 5.0);
}
float lightningStrength(float t){
    float seg = floor(t * 2.0);
    float frac = fract(t * 2.0);
    // gate fires only when this segment's seed AND the drums punch are both high
    float trigger = step(0.85, hash11(seg*0.91));
    float gate    = trigger * smoothstep(0.25, 0.7, u_drumsPunch);
    // sharp double-flash decay
    float flash   = exp(-frac * 6.0) * (1.0 + 0.5*exp(-(frac-0.08)*(frac-0.08)*200.0));
    return flash * gate;
}

vec4 marchClouds(vec3 ro, vec3 rd, float tMin, float tMax, int steps, float t){
    float dt = (tMax - tMin) / float(steps);
    vec3 sunDir = normalize(vec3(0.4, 0.7, -0.5));
    vec3 sunCol = vec3(0.95, 0.85, 0.70);
    vec3 ambCol = vec3(0.10, 0.13, 0.20);

    vec3 boltP = lightningPos(t);
    float boltI = lightningStrength(t);
    vec3 boltCol = vec3(0.55, 0.7, 1.0) * boltI * 2.0;

    float tCur = tMin + dt * hash12(rd.xy + t);  // jitter to break up banding
    vec3 col = vec3(0.0);
    float trans = 1.0;
    for (int i = 0; i < 96; i++){
        if (i >= steps) break;
        if (trans < 0.02) break;
        vec3 p = ro + rd * tCur;
        float d = cloudDensity(p, t);
        if (d > 0.0){
            // single short shadow ray toward the sun (cheap self-shadow)
            float shadow = 0.0;
            for (int j = 0; j < 3; j++){
                vec3 sp = p + sunDir * (0.08 * float(j+1));
                shadow += cloudDensity(sp, t);
            }
            float sunT = exp(-shadow * 2.0);

            // bolt contribution: 1/r^2 light from current bolt position
            vec3 boltVec = boltP - p;
            float bd = length(boltVec);
            float boltShade = boltI / (bd*bd + 1.0);

            vec3 lit = ambCol + sunCol * sunT + boltCol * boltShade;
            float alpha = 1.0 - exp(-d * dt * 5.0);
            col += trans * alpha * lit;
            trans *= 1.0 - alpha;
        }
        tCur += dt;
    }
    return vec4(col, 1.0 - trans);
}

// =====================================================================
// MODE 3 — VOLUMETRIC CLOUDS + LIGHTNING  (main / drive)
// Camera looking forward into a cloud field. Drum punches gate lightning.
// =====================================================================
// Intersect ray (origin, dir) with horizontal slab y in [yLo, yHi].
// Returns vec2(tMin, tMax); tMax<=tMin means no hit.
vec2 slabHit(vec3 ro, vec3 rd, float yLo, float yHi){
    float t0 = (yLo - ro.y) / rd.y;
    float t1 = (yHi - ro.y) / rd.y;
    float tMin = min(t0, t1);
    float tMax = max(t0, t1);
    tMin = max(tMin, 0.0);
    return vec2(tMin, tMax);
}

vec3 mode_clouds(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time;

    // camera below the cloud slab, looking forward + slightly up
    vec3 ro = vec3(0.0, 0.05, -2.0 + t*0.06);
    vec3 rd = normalize(vec3(p, 1.1));
    rd.yz = rot2(0.18 + u_bass*0.05) * rd.yz;     // tilt up into the clouds

    // sky / horizon backdrop
    float horizon = smoothstep(-0.05, 0.5, rd.y);
    vec3 sky = mix(vec3(0.04, 0.06, 0.10), vec3(0.10, 0.18, 0.30), horizon);
    sky += vec3(0.6, 0.45, 0.35) * pow(max(1.0 - horizon, 0.0), 6.0) * 0.4;  // warm horizon

    // ground / dark below the camera
    if (rd.y < 0.0){
        sky = mix(vec3(0.02, 0.025, 0.04), sky, smoothstep(-0.4, 0.0, rd.y));
    }

    // bright lightning flash bleeds through the sky too
    float boltI = lightningStrength(t);
    sky += vec3(0.5, 0.65, 0.95) * boltI * 0.35;

    // only march the cloud slab if the ray enters it
    vec2 hit = slabHit(ro, rd, SLAB_LO, SLAB_HI);
    vec3 col = sky;
    if (hit.y > hit.x && rd.y > 0.0){
        float tMin = hit.x;
        float tMax = min(hit.y, 14.0);
        vec4 cl = marchClouds(ro, rd, tMin, tMax, 56, t);
        col = cl.rgb + sky * (1.0 - cl.a);
    }

    // global flash burst on lightning
    col += vec3(0.4, 0.55, 0.85) * boltI * 0.25;

    // bass adds a low rumble glow at the bottom
    col += vec3(0.08, 0.10, 0.20) * u_bass * 0.25 * smoothstep(0.0, -0.4, p.y);

    return col;
}

// =====================================================================
// MODE 4 — STARFIELD + LENSED BLACK HOLE  (breakdown / pull-back)
// Background star field, gravitational-lens distortion toward centre,
// thin accretion disk that pulses with the beat.
// =====================================================================
vec3 starField(vec2 uv){
    vec3 col = vec3(0.0);
    // three layers at different scales to fake parallax
    for (int k = 0; k < 3; k++){
        float fk = float(k);
        float scale = mix(120.0, 700.0, fk/2.0);
        vec2 g = uv * scale;
        vec2 gi = floor(g);
        vec2 gf = fract(g) - 0.5;
        float h = hash12(gi + fk*17.7);
        float thr = mix(0.991, 0.998, fk/2.0);
        if (h > thr){
            float bright = (h - thr) / (1.0 - thr);
            float spark = exp(-dot(gf, gf) * mix(40.0, 200.0, fk/2.0));
            // tint by sub-hash — slight blue/orange variety
            float tint = hash12(gi + fk*3.1);
            vec3 c = mix(vec3(0.7, 0.85, 1.05), vec3(1.05, 0.9, 0.75), tint);
            col += c * spark * bright * mix(1.2, 0.5, fk/2.0);
        }
    }
    // faint dust band
    float dust = fbm2(uv*4.0 + 11.0);
    col += vec3(0.05, 0.06, 0.10) * pow(dust, 3.0);
    return col;
}

vec3 mode_blackhole(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time;

    float r = length(p);
    float ang = atan(p.y, p.x);

    // event horizon size, swells with bass
    float eh = 0.10 + u_bass * 0.04;

    // gravitational lensing: warp the angle of arrival of background light
    // pull radial sample inward by ~k/r^2
    float k = 0.06 + u_bassPunch*0.04;
    float warpedR = max(r - k / max(r, 0.01), 0.001);
    // also swirl with frame-dragging proportional to 1/r
    float swirl = 0.8 / max(r, 0.05) + t*0.05;
    vec2 sampleP = vec2(cos(ang + swirl), sin(ang + swirl)) * warpedR;
    sampleP = sampleP * 0.5 + 0.5;        // back to uv space

    vec3 stars = starField(sampleP);

    // inside the event horizon: pure void
    if (r < eh) return vec3(0.0);

    // accretion disk — thin glowing annulus, warm reddish-orange always.
    // Centroid only nudges within the warm range; never sliding to green.
    float diskInner = eh * 1.25;
    float diskOuter = eh * 2.4;
    // soft inner edge, sharp outer edge for that "edge-on disk" feel
    float diskMask = smoothstep(diskInner, diskInner + 0.012, r) *
                     smoothstep(diskOuter, diskOuter - 0.05, r);
    // hot-spot variation around the ring, advected by time + beat
    float spots = 0.5 + 0.5 * sin(ang*6.0 + t*1.4 + u_drumsPunch*3.0);
    spots *= 0.5 + 0.5*fbm2(vec2(ang*3.0, r*20.0 + t*0.6));
    float hue = fract(0.02 + 0.05*u_centroid);                      // red→orange
    vec3 diskCol = hsv(vec3(hue, 0.85, 1.0)) * (0.4 + 0.9*spots);
    // doppler-ish brightening on one side
    diskCol *= 0.5 + 0.7 * (0.5 + 0.5*cos(ang));

    vec3 col = stars;
    col += diskCol * diskMask * (0.6 + u_globalRms*0.7);

    // photon ring — sharp bright thin annulus right at the event horizon
    float photon = exp(-pow((r - eh*1.08) / 0.005, 2.0));
    col += vec3(1.05, 0.85, 0.6) * photon * (0.5 + u_drumsPunch*0.9);

    // very soft outer halo
    col += diskCol * 0.08 * exp(-pow((r - eh*2.6)*3.5, 2.0));

    return col;
}

// =====================================================================
// MODE 5 — AURORA OVER CLOUDS  (outro / rise)
// Cloud volume below + procedural aurora curtains above + horizon glow
// that grows over section_t.
// =====================================================================
float auroraCurtain(vec2 p, float t, float seed){
    // vertical curtains: shape = horizontal sin warped by FBM
    float warp = fbm2(p*vec2(0.6, 1.8) + vec2(t*0.05 + seed, t*0.02));
    float band = sin((p.x*1.4 + warp*1.6 + seed)*3.0 + t*0.4) * 0.5 + 0.5;
    band = pow(band, 5.0);
    // streak the curtain vertically with a second high-freq term so it has
    // visible ribbons rather than uniform glow
    float streak = 0.5 + 0.5*sin(p.y*40.0 + warp*8.0 + seed*7.0);
    band *= 0.4 + 0.6*streak;
    // vertical falloff: brightest mid-altitude, hard top cutoff
    float vf = smoothstep(0.42, 0.58, p.y) * smoothstep(0.95, 0.72, p.y);
    return band * vf;
}

vec3 mode_aurora(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time;

    // base sky gradient — deep night up top, warm horizon glow growing with sectionT
    float sectionLift = smoothstep(0.0, 1.0, u_sectionT);
    vec3 night = vec3(0.01, 0.02, 0.05);
    vec3 glow  = vec3(0.30, 0.12, 0.18) * (0.2 + sectionLift*0.6);
    vec3 sky = mix(glow, night, smoothstep(-0.15, 0.4, p.y));

    // a few stars in the upper sky
    if (p.y > 0.15){
        vec2 sp = floor((p + vec2(2.0, 0.0)) * 240.0);
        float s = step(0.997, hash12(sp));
        sky += vec3(0.7, 0.85, 1.0) * s * smoothstep(0.15, 0.7, p.y) * 0.7;
    }

    // aurora ribbons — three offset curtains in green/violet
    vec2 ap = vec2(uv.x*2.0 - 1.0, uv.y);
    float a1 = auroraCurtain(ap*vec2(2.0, 1.0),       t, 0.0);
    float a2 = auroraCurtain(ap*vec2(2.7, 1.1) + 1.3, t, 4.7);
    float a3 = auroraCurtain(ap*vec2(1.6, 0.9) - 0.7, t, 9.2);
    vec3 auroraCol =
        vec3(0.25, 1.10, 0.55) * a1 +
        vec3(0.40, 0.85, 1.20) * a2 * 0.85 +
        vec3(0.85, 0.40, 1.10) * a3 * 0.7;
    // music response — the curtains pulse with vocals + other
    auroraCol *= 0.6 + 1.4*(u_vocalsRms*0.7 + u_otherRms*0.5);

    sky += auroraCol;

    // clouds below the horizon — short cheap march, only for the lower band
    if (uv.y < 0.5){
        vec3 ro = vec3(0.0, 0.05, -2.0 + t*0.05);
        vec3 rd = normalize(vec3(p, 1.1));
        rd.yz = rot2(0.10) * rd.yz;
        vec2 hit = slabHit(ro, rd, SLAB_LO, SLAB_HI);
        if (hit.y > hit.x && rd.y > 0.0){
            vec4 cl = marchClouds(ro, rd, hit.x, min(hit.y, 12.0), 36, t);
            // tint clouds dark and cool — they should mostly be silhouettes
            vec3 cloudTint = mix(vec3(0.20, 0.22, 0.30), vec3(0.45, 0.30, 0.35), sectionLift);
            cl.rgb *= cloudTint;
            sky = sky * (1.0 - cl.a) + cl.rgb;
        }
    }

    // tight horizon line glow (rises with section)
    float horizonBand = exp(-pow((uv.y - 0.42)*40.0, 2.0));
    sky += vec3(0.9, 0.45, 0.40) * horizonBand * sectionLift * (0.5 + u_globalRms*0.4);

    return sky;
}

// =====================================================================
// DISPATCH
// =====================================================================
vec3 dispatch(int mode, vec2 uv){
    if (mode == 0) return mode_water(uv);
    if (mode == 1) return mode_ridges(uv);
    if (mode == 2) return mode_plasma(uv);
    if (mode == 3) return mode_clouds(uv);
    if (mode == 4) return mode_blackhole(uv);
    return mode_aurora(uv);
}

void main(){
    vec3 col = dispatch(u_modeA, v_uv);
    // skip the second mode entirely when not crossfading — saves up to 2x cost
    if (u_modeBlend > 0.001){
        vec3 b = dispatch(u_modeB, v_uv);
        col = mix(col, b, smoothstep(0.0, 1.0, u_modeBlend));
    }

    // gentle global beat-phase brightness lift (subtle)
    col *= 1.0 + 0.05 * (1.0 - u_beatPhase);

    frag = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Bright extract for bloom.
# Higher threshold + sharper cutoff so highlights pop instead of fogging.
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
    float w = smoothstep(u_threshold, u_threshold + 0.25, b);
    // square the weight to keep dim mids out of the bloom
    frag = vec4(c * w * w, 1.0);
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

// Hash without sine — same Hoskins hash used in the scene shader. The sin-based
// hash this used to use produced visible diagonal banding; this one is uniform.
float hash12g(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

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

    // film grain — triangular PDF (sum of two uniforms - 1) gives a much more
    // organic look than a single uniform, and the Hoskins hash keeps the
    // pattern truly isotropic (no diagonal moiré).
    vec2 gp = v_uv * vec2(1920.0, 1080.0);
    float r1 = hash12g(gp + u_time*60.0);
    float r2g = hash12g(gp + u_time*60.0 + 113.7);
    float g = (r1 + r2g) - 1.0;       // [-1, 1] triangular
    col += vec3(g) * u_grainAmt;

    // gamma to sRGB
    col = pow(max(col, 0.0), vec3(1.0/2.2));
    frag = vec4(col, 1.0);
}
"""
