"""GLSL shader sources for the offline music visualizer.

Pipeline:
  scene_fs   -> HDR float texture (the actual visual)
  bright_fs  -> extracts bright pixels for bloom
  blur_fs    -> separable gaussian, run H then V (3 mip levels)
  composite_fs -> tonemap + bloom add + chromatic aberration + grain + vignette

Mode line-up:
  0 intro      original cyan fog with vertical light shafts
  1 verse      "Protean clouds" volumetric flow noise (Nimitz)
  2 drop       original magenta shockwave
  3 main       cloud + ray-marched lightning bolts (al-ro)
  4 breakdown  domain-warped FBM colormap (Iquilezles warp)
  5 outro      "Seascape" raymarched water (Alexander Alekseev)

Donor shader functions are kept prefixed (pc_*, cl_*, wf_*, sea_*) to avoid
name collisions across modes. Music response is added on top of the donor
shaders rather than baked in, so the visual base stays close to source.
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

uniform float u_bass;
uniform float u_bassPunch;
uniform float u_drumsRms;
uniform float u_drumsPunch;
uniform float u_vocalsRms;
uniform float u_otherRms;
uniform float u_globalRms;
uniform float u_centroid;
uniform float u_pitch;
uniform float u_beatPhase;
uniform float u_chroma[12];

const float PI  = 3.14159265359;
const float TAU = 6.28318530718;

mat2 rot2(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

// -----------------------------------------------------------------
// Hoskins hashes (no diagonal banding).
// -----------------------------------------------------------------
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

// 2D value noise / fbm used by the original intro mode and the dissolve mask.
float vnoise2(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash12(i), b = hash12(i+vec2(1,0));
    float c = hash12(i+vec2(0,1)), d = hash12(i+vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm2(vec2 p){
    float v = 0.0, a = 0.5;
    for (int i=0; i<5; i++){ v += a*vnoise2(p); p = rot2(0.7)*p*2.03; a *= 0.5; }
    return v;
}
// domain-warped fbm — original intro shader's "warpedFbm".
float warpedFbm(vec2 p, float t){
    vec2 q = vec2(fbm2(p + vec2(0.0, t*0.15)), fbm2(p + vec2(5.2, t*0.12)));
    vec2 r = vec2(fbm2(p + 4.0*q + vec2(1.7, 9.2) + t*0.08),
                  fbm2(p + 4.0*q + vec2(8.3, 2.8) + t*0.07));
    return fbm2(p + 4.0*r);
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

// =================================================================
// MODE 0 — INTRO (original "cryogenic" cyan fog with light shafts)
// =================================================================
vec3 mode_intro(vec2 uv){
    vec2 p = uv * vec2(u_res.x/u_res.y, 1.0);
    float t = u_time * 0.08;
    float n = warpedFbm(p*1.6 + vec2(t, -t*0.3), u_time);
    n = pow(n, 1.4);

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

    col *= mix(0.6, 1.1, smoothstep(0.0, 0.9, u_sectionT));
    return col;
}

// =================================================================
// MODE 1 — PROTEAN CLOUDS  (Nimitz "Protean clouds")
// Volumetric raymarched grid-noise. Music is wired in via:
//   prm1 (morph factor)   <- u_centroid + bass
//   bsMo (camera offset)  <- u_bassPunch
//   forward speed         <- u_globalRms
// =================================================================
const mat3 PC_M3 = mat3(0.33338, 0.56034, -0.71817,
                       -0.87887, 0.32651, -0.15323,
                        0.15162, 0.69596,  0.61339)*1.93;

float pc_mag2(vec2 p){ return dot(p,p); }
float pc_linstep(float mn, float mx, float x){ return clamp((x-mn)/(mx-mn), 0.0, 1.0); }
mat2  pc_rot(float a){ float c=cos(a),s=sin(a); return mat2(c,s,-s,c); }
vec2  pc_disp(float t){ return vec2(sin(t*0.22), cos(t*0.175))*2.0; }

// Globals replicated as args because GLSL has no mutable globals across funcs.
vec2 pc_map(vec3 p, float prm1, float bsMoY){
    vec3 p2 = p;
    p2.xy -= pc_disp(p.z).xy;
    p.xy *= pc_rot(sin(p.z+u_time)*(0.1 + prm1*0.05) + u_time*0.09);
    float cl = pc_mag2(p2.xy);
    float d = 0.0;
    p *= 0.61;
    float z = 1.0;
    float trk = 1.0;
    float dspAmp = 0.1 + prm1*0.2;
    for(int i = 0; i < 5; i++){
        p += sin(p.zxy*0.75*trk + u_time*trk*0.8)*dspAmp;
        d -= abs(dot(cos(p), sin(p.yzx))*z);
        z *= 0.57;
        trk *= 1.4;
        p = p*PC_M3;
    }
    d = abs(d + prm1*3.0) + prm1*0.3 - 2.5 + bsMoY;
    return vec2(d + cl*0.2 + 0.25, cl);
}

vec4 pc_render(vec3 ro, vec3 rd, float time, float prm1, vec2 bsMo){
    vec4 rez = vec4(0);
    const float ldst = 8.0;
    float t = 1.5;
    float fogT = 0.0;
    for(int i=0; i<110; i++){
        if(rez.a > 0.99) break;
        vec3 pos = ro + t*rd;
        vec2 mpv = pc_map(pos, prm1, bsMo.y);
        float den = clamp(mpv.x-0.3, 0.0, 1.0)*1.12;
        float dn = clamp(mpv.x + 2.0, 0.0, 3.0);
        vec4 col = vec4(0);
        if(mpv.x > 0.6){
            col = vec4(sin(vec3(5.0,0.4,0.2) + mpv.y*0.1 + sin(pos.z*0.4)*0.5 + 1.8)*0.5 + 0.5, 0.08);
            col *= den*den*den;
            col.rgb *= pc_linstep(4.0, -2.5, mpv.x)*2.3;
            float dif  = clamp((den - pc_map(pos+0.8, prm1, bsMo.y).x)/9.0,  0.001, 1.0);
            dif       += clamp((den - pc_map(pos+0.35, prm1, bsMo.y).x)/2.5, 0.001, 1.0);
            col.xyz *= den*(vec3(0.005,0.045,0.075) + 1.5*vec3(0.033,0.07,0.03)*dif);
        }
        float fogC = exp(t*0.2 - 2.2);
        col.rgba += vec4(0.06,0.11,0.11, 0.1)*clamp(fogC-fogT, 0.0, 1.0);
        fogT = fogC;
        rez = rez + col*(1.0 - rez.a);
        t += clamp(0.5 - dn*dn*0.05, 0.09, 0.3);
    }
    return clamp(rez, 0.0, 1.0);
}

float pc_getsat(vec3 c){
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi)/(ma + 1e-7);
}
vec3 pc_iLerp(vec3 a, vec3 b, float x){
    vec3 ic = mix(a, b, x) + vec3(1e-6, 0.0, 0.0);
    float sd = abs(pc_getsat(ic) - mix(pc_getsat(a), pc_getsat(b), x));
    vec3 dir = normalize(vec3(2.0*ic.x - ic.y - ic.z,
                              2.0*ic.y - ic.x - ic.z,
                              2.0*ic.z - ic.y - ic.x));
    float lgt = dot(vec3(1.0), ic);
    float ff = dot(dir, normalize(ic));
    ic += 1.5*dir*sd*ff*lgt;
    return clamp(ic, 0.0, 1.0);
}

vec3 mode_protean(vec2 uv){
    vec2 q = uv;
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);

    // Donor's natural prm1 oscillation kept; music nudges (a touch faster
    // morph on energy, slight phase push from centroid).
    float morphPhase = u_time*(0.30 + 0.20*u_globalRms) + u_centroid*1.6;
    float prm1 = smoothstep(-0.4, 0.4, sin(morphPhase));
    // bass punch pushes the camera offset, which displaces the volume centre
    vec2 bsMo = vec2(0.0, -0.05 - 0.45*u_bassPunch);

    float speedBoost = 1.0 + 1.4*u_globalRms;
    float time = u_time*3.0*speedBoost;

    vec3 ro = vec3(0,0,time);
    ro += vec3(sin(u_time)*0.5, 0.0, 0.0);
    float dspAmp = 0.85;
    ro.xy += pc_disp(ro.z)*dspAmp;
    float tgtDst = 3.5;

    vec3 target = normalize(ro - vec3(pc_disp(time + tgtDst)*dspAmp, time + tgtDst));
    ro.x -= bsMo.x*2.0;
    vec3 rightdir = normalize(cross(target, vec3(0,1,0)));
    vec3 updir = normalize(cross(rightdir, target));
    rightdir = normalize(cross(updir, target));
    vec3 rd = normalize((p.x*rightdir + p.y*updir)*1.0 - target);
    rd.xy *= pc_rot(-pc_disp(time + 3.5).x*0.2 + bsMo.x);

    vec4 scn = pc_render(ro, rd, time, prm1, bsMo);
    vec3 col = scn.rgb;
    col = pc_iLerp(col.bgr, col.rgb, clamp(1.0 - prm1, 0.05, 1.0));
    // Apply donor's grading (matches the original Shadertoy look), then
    // linearize so our composite's sRGB encode doesn't double-gamma it.
    col = pow(col, vec3(0.55, 0.65, 0.6))*vec3(1.0, 0.97, 0.9);
    col = pow(max(col, 0.0), vec3(2.2));
    return col;
}

// =================================================================
// MODE 2 — DROP (original magenta shockwave + particles)
// =================================================================
float drop_hash(vec2 p){ return hash12(p); }
vec3 mode_drop(vec2 uv){
    vec2 p = (uv - 0.5) * vec2(u_res.x/u_res.y, 1.0);
    float r = length(p);
    float ang = atan(p.y, p.x);

    float wave = 0.0;
    for (int i=0; i<3; i++){
        float fi = float(i);
        float t = fract(u_time*0.4 + fi*0.33);
        float radius = t * 1.2;
        wave += exp(-pow((r - radius)*15.0, 2.0)) * (1.0 - t) * (0.6 + u_bass*1.5);
    }

    float parts = 0.0;
    for (int i=0; i<6; i++){
        float fi = float(i);
        float seed = drop_hash(vec2(fi, 7.0));
        float a = seed * TAU + sin(u_time*0.3 + fi)*0.4;
        float life = fract(u_time*0.25 + seed);
        vec2 pos = vec2(cos(a), sin(a)) * life * 1.5;
        float d = length(p - pos);
        parts += exp(-d*40.0) * (1.0 - life) * (0.5 + u_drumsPunch);
    }

    float spiral = 0.5 + 0.5*sin(ang*6.0 + r*15.0 - u_time*2.0);
    vec3 bg = mix(vec3(0.05,0.0,0.1), vec3(0.4,0.05,0.5), spiral);

    vec3 col = bg;
    col += vec3(1.0, 0.4, 0.9) * wave;
    col += vec3(1.0, 0.95, 1.0) * parts * 1.5;
    col += vec3(0.9, 0.2, 0.7) * u_bassPunch * 0.5;
    return col;
}

// =================================================================
// MODE 3 — VOLUMETRIC CLOUDS + RAY-MARCHED LIGHTNING (al-ro)
// Music wiring:
//   strikeFrequency       <- u_drumsPunch
//   bolt impact radius    <- u_bassPunch
//   internalFrequency     <- u_drumsPunch
//   density multiplier    <- u_otherRms
//   sun direction sway    <- u_bass
// =================================================================
const float CL_CLOUD_START = 20.0;
const float CL_CLOUD_HEIGHT = 20.0;
const float CL_CLOUD_END = 40.0;
const float CL_CLOUD_EXTENT = 20.0;
const int   CL_MAX_STEPS = 32;
const float CL_MIN_DIST = 0.1;
const float CL_MAX_DIST = 1000.0;
const float CL_EPSILON = 1e-4;
const int   CL_STEPS_PRIMARY = 24;
const int   CL_STEPS_LIGHT = 6;
const vec3  CL_BOLT_COLOUR = vec3(0.30, 0.6, 1.0);
const vec3  CL_HORIZON = vec3(1.0, 0.9, 0.8);
const vec3  CL_SKY = vec3(0.0);
const vec3  CL_SUNCOL = vec3(1.0);
const float CL_GOLDEN = 1.61803398875;

vec3 cl_minCorner = vec3(-CL_CLOUD_EXTENT, CL_CLOUD_START, -CL_CLOUD_EXTENT);
vec3 cl_maxCorner = vec3( CL_CLOUD_EXTENT, CL_CLOUD_END,    CL_CLOUD_EXTENT);

float cl_saturate(float x){ return clamp(x, 0.0, 1.0); }
float cl_remap(float x, float a, float b, float c, float d){ return c + (x-a)*(d-c)/(b-a); }

float cl_hash1(float p){
    vec3 p3 = fract(vec3(p) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 cl_hash33(vec3 p3){
    p3 = fract(p3 * vec3(0.1031, 0.11369, 0.13787));
    p3 += dot(p3, p3.yxz + 19.19);
    return -1.0 + 2.0*fract(vec3((p3.x+p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}
vec3 cl_hash31(float p){
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz)*p3.zyx);
}

float cl_fade(float t){ return t*t*t*(t*(6.0*t-15.0)+10.0); }
float cl_grad(float h, float p){ int i = int(1e4*h); return ((i & 1)==0) ? p : -p; }
float cl_perlin1(float p){
    float pi = floor(p), pf = p - pi, w = cl_fade(pf);
    return mix(cl_grad(cl_hash1(pi), pf),
               cl_grad(cl_hash1(pi+1.0), pf-1.0), w) * 2.0;
}
float cl_perlin3(vec3 p){
    vec3 pi = floor(p), pf = p - pi;
    vec3 w = pf*pf*(3.0-2.0*pf);
    return mix(
        mix(
            mix(dot(pf-vec3(0,0,0), cl_hash33(pi+vec3(0,0,0))),
                dot(pf-vec3(1,0,0), cl_hash33(pi+vec3(1,0,0))), w.x),
            mix(dot(pf-vec3(0,0,1), cl_hash33(pi+vec3(0,0,1))),
                dot(pf-vec3(1,0,1), cl_hash33(pi+vec3(1,0,1))), w.x),
            w.z),
        mix(
            mix(dot(pf-vec3(0,1,0), cl_hash33(pi+vec3(0,1,0))),
                dot(pf-vec3(1,1,0), cl_hash33(pi+vec3(1,1,0))), w.x),
            mix(dot(pf-vec3(0,1,1), cl_hash33(pi+vec3(0,1,1))),
                dot(pf-vec3(1,1,1), cl_hash33(pi+vec3(1,1,1))), w.x),
            w.z),
        w.y);
}

bool cl_intersectPlane(vec3 n, vec3 pp, vec3 org, vec3 dir, out float t){
    float denom = dot(n, dir);
    if (denom > 1e-6){ t = dot(pp-org, n)/denom; return t >= 0.0; }
    return false;
}

float cl_sdCappedCylinder(vec3 p, float h, float r){
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(h, r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float cl_fbm(float pos, int octaves){
    if (pos < 0.0) return 0.0;
    float total = 0.0;
    float frequency = 0.2;
    float amplitude = 1.0;
    for (int i = 0; i < 8; i++){
        if (i >= octaves) break;
        float p = pos;
        if (i > 2) p += 0.5*u_time;
        total += cl_perlin1(p * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return total;
}

float cl_getGlow(float dist, float radius, float intensity){
    dist = max(dist, 1e-6);
    return pow(radius/dist, intensity);
}

// Three independent bolts. Returns SDF; outputs xz of each bolt's exit point.
float cl_getSDF(vec3 p, out vec2 b0, out vec2 b1, out vec2 b2){
    float dist = 1e10;
    p.y -= CL_CLOUD_START;
    float strikeFreq = 0.10 + 0.55*u_drumsPunch;     // music boost
    float speed = 2.5;
    float boltLength = CL_CLOUD_START * 0.5;
    float range = CL_CLOUD_EXTENT * 0.4;
    float scale = 0.5;
    float descentDuration = 0.5;
    float radiusBase = 0.01 + 0.03*u_bassPunch;
    int octaves = 4;
    b0 = vec2(1e10); b1 = vec2(1e10); b2 = vec2(1e10);
    float shift = 0.0;
    float shapeOffset = 15.2;

    for (int i = 0; i < 3; i++){
        shapeOffset *= 2.0;
        shift = fract(shift + 0.25);
        float time = u_time*speed + shift;
        float t = floor(time) + 1.0;

        if (cl_hash1(float(i) + t*0.026) > strikeFreq) continue;

        vec2 location = 2.0*vec2(cl_hash1(t+float(i)+0.43), cl_hash1(t+float(i)+0.3))-1.0;
        location *= range;
        float progress = clamp(fract(time)/descentDuration, 0.0, 1.0);
        float radius = radiusBase;
        if (progress > 0.95 && fract(time) - descentDuration < 0.1){
            radius = 0.10 + 0.06*u_bassPunch;
        }
        progress *= boltLength;
        vec3 offset = vec3(location.x + cl_fbm(shapeOffset+t*0.20+(scale*p.y), octaves),
                           progress,
                           location.y + cl_fbm(shapeOffset+t*0.12-(scale*p.y), octaves));

        if (i == 0) b0 = -location.xy;
        if (i == 1) b1 = -location.xy;
        if (i == 2) b2 = -location.xy;
        dist = min(dist, cl_sdCappedCylinder(p+offset, radius, progress));
    }
    return dist;
}

float cl_distanceToScene(vec3 ro, vec3 rd, float startD, float endD,
                         out vec3 glow, out vec2 b0, out vec2 b1, out vec2 b2){
    float depth = startD;
    float dist;
    glow = vec3(0.0);
    b0 = vec2(1e10); b1 = vec2(1e10); b2 = vec2(1e10);
    for (int i = 0; i < CL_MAX_STEPS; i++){
        vec3 pp = ro + depth*rd;
        dist = 0.5 * cl_getSDF(pp, b0, b1, b2);
        glow += cl_getGlow(dist, 0.01, 0.8) * CL_BOLT_COLOUR;
        if (dist < CL_EPSILON) return depth;
        depth += dist;
        if (depth >= endD) return endD;
    }
    return endD;
}

vec3 cl_getSkyColour(vec3 rd){
    if (rd.y < 0.0) return vec3(0.025);
    return mix(CL_HORIZON, CL_SKY, pow(rd.y, 0.03));
}

vec2 cl_intersectAABB(vec3 ro, vec3 rd, vec3 mn, vec3 mx){
    vec3 tMin = (mn - ro)/rd;
    vec3 tMax = (mx - ro)/rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}
bool cl_inside(vec3 p){
    float e = 1e-4;
    return all(greaterThan(p, cl_minCorner-e)) && all(lessThan(p, cl_maxCorner+e));
}
bool cl_getCloudIntersection(vec3 ro, vec3 dir, out float dStart, out float dTotal){
    vec2 it = cl_intersectAABB(ro, dir, cl_minCorner, cl_maxCorner);
    if (cl_inside(ro)) it.x = 1e-4;
    dStart = it.x;
    dTotal = it.y - it.x;
    return it.x > 0.0 && it.x < it.y;
}

float cl_getNoise(vec3 pos, float speed){
    return 0.5 + 0.5*cl_perlin3(speed*u_time + pos);
}

float cl_clouds(vec3 p, out float cloudHeight){
    cloudHeight = cl_saturate((p.y - CL_CLOUD_START)/(CL_CLOUD_END - CL_CLOUD_START));

    float bottom = 1.0 - cl_saturate(length(p.xz)/(1.25*CL_CLOUD_EXTENT));
    bottom *= cl_saturate(cl_remap(cloudHeight, 0.25*bottom, 1.0, 1.0, 0.0))
            * cl_saturate(cl_remap(cloudHeight, 0.0, 0.175, 0.45, 1.0));
    bottom = cl_saturate(cl_remap(bottom, 0.5*cl_getNoise(0.25*p, 0.05), 1.0, 0.0, 1.0));
    bottom = cl_saturate(cl_remap(bottom, 0.15*cl_getNoise(1.0*p, 0.20), 1.0, 0.0, 1.0));

    float top = 1.0 - cl_saturate(length(p.xz)/(1.5*CL_CLOUD_EXTENT));
    top *= cl_saturate(cl_remap(1.0-cloudHeight, 0.25*top, 1.0, 1.0, 0.0))
         * cl_saturate(cl_remap(1.0-cloudHeight, 0.0, 0.175, 0.45, 1.0));
    top = cl_saturate(cl_remap(top, 0.5*cl_getNoise(0.25*p, 0.05), 1.0, 0.0, 1.0));
    top = cl_saturate(cl_remap(top, 0.15*cl_getNoise(1.0*p, 0.20), 1.0, 0.0, 1.0));

    float densityMul = 6.5 + 1.5*u_otherRms;
    return (bottom + top) * densityMul;
}

float cl_HG(float g, float costh){
    return (1.0/(4.0*PI)) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0*g*costh, 1.5));
}

float cl_lightRay(vec3 p, float mu, vec3 sunDir){
    float lightRayDistance = CL_CLOUD_EXTENT * 0.75;
    float distToStart = 0.0;
    cl_getCloudIntersection(p, sunDir, distToStart, lightRayDistance);
    float stepL = lightRayDistance/float(CL_STEPS_LIGHT);
    float density = 0.0;
    float ch = 0.0;
    for (int j = 0; j < CL_STEPS_LIGHT; j++){
        density += mix(1.0, 0.75, mu) *
                   cl_clouds(p + sunDir*float(j)*stepL, ch);
    }
    float beers = max(exp(-stepL*density),
                      exp(-stepL*density*0.2)*0.75);
    return mix(beers*2.0*(1.0-exp(-stepL*density*2.0)), beers, mu);
}

vec3 cl_mainRay(vec3 ro, vec3 dir, vec3 sunDir, out float totalT, float mu, float offset, inout float dCloud,
                vec2 b0, vec2 b1, vec2 b2){
    totalT = 1.0;
    vec3 col = vec3(0.0);
    float dStart = 0.0, dTotal = 0.0;
    if (!cl_getCloudIntersection(ro, dir, dStart, dTotal)) return col;
    float stepS = dTotal/float(CL_STEPS_PRIMARY);
    dStart += stepS*offset;
    float dist = dStart;
    vec3 p = ro + dist*dir;
    float phase = mix(cl_HG(-0.3, mu), cl_HG(0.3, mu), 0.7);
    float power = 6.0;
    vec3 sunLight = CL_SUNCOL * power;
    float internalFreq = 0.50 - 0.40*u_drumsPunch;   // lower threshold = more flicker

    for (int i = 0; i < CL_STEPS_PRIMARY; i++){
        float ch;
        float density = cl_clouds(p, ch);
        float sigmaS = 1.0;
        float sampleSigmaS = sigmaS*density;
        float sampleSigmaE = sampleSigmaS;
        if (density > 0.0){
            dCloud = min(dCloud, dist);
            // internal flicker source
            vec3 source = vec3(0, CL_CLOUD_START + CL_CLOUD_HEIGHT*0.5, 0)
                        + (2.0*cl_hash31(floor(u_time*5.0)) - 1.0) * CL_CLOUD_EXTENT*0.25;
            float prox = length(p - source);
            float sz = sin(45.0*fract(u_time)) + 5.0;
            vec3 internal = cl_getGlow(prox, sz, 3.2) * CL_BOLT_COLOUR;
            if (cl_hash1(floor(u_time)) > internalFreq) internal = vec3(0);
            // exit-point ambient at cloud bottom
            float h = 0.9*CL_CLOUD_START;
            float ssz = 3.0;
            internal += cl_getGlow(length(p - vec3(b0.x, h, b0.y)), ssz, 2.2) * CL_BOLT_COLOUR;
            internal += cl_getGlow(length(p - vec3(b1.x, h, b1.y)), ssz, 2.2) * CL_BOLT_COLOUR;
            internal += cl_getGlow(length(p - vec3(b2.x, h, b2.y)), ssz, 2.2) * CL_BOLT_COLOUR;

            vec3 ambient = internal + CL_SUNCOL * mix(0.05, 0.125, ch);
            vec3 lum = ambient + sunLight*phase*cl_lightRay(p, mu, sunDir);
            lum *= sampleSigmaS;
            float trn = exp(-sampleSigmaE*stepS);
            col += totalT*(lum - lum*trn)/sampleSigmaE;
            totalT *= trn;
            if (totalT <= 0.01){ totalT = 0.0; return col; }
        }
        dist += stepS;
        p = ro + dir*dist;
    }
    return col;
}

vec3 mode_clouds(vec2 uv){
    vec2 fragCoord = uv * u_res;
    vec3 rd0;
    {
        // 40 deg field of view
        vec2 xy = fragCoord - u_res*0.5;
        float z = (0.5*u_res.y) / tan(radians(40.0)*0.5);
        rd0 = normalize(vec3(xy, -z));
    }

    // Camera flies in a slow horizontal arc 6 units high, looking at the cloud center.
    // Bass adds a subtle vertical sway.
    float ct = u_time*0.07;
    vec3 ro = vec3(sin(ct)*8.0, 6.0 + sin(u_time*0.4)*0.4*u_bass, -38.0 + cos(ct)*4.0);
    vec3 target = vec3(0.0, CL_CLOUD_START + CL_CLOUD_HEIGHT*0.5, 0.0) - ro;

    vec3 zaxis = normalize(target);
    vec3 xaxis = normalize(cross(zaxis, vec3(0,1,0)));
    vec3 yaxis = cross(xaxis, zaxis);
    mat3 view = mat3(xaxis, yaxis, -zaxis);
    vec3 rd = normalize(view * rd0);

    // Hash dither replaces blue-noise texture; golden-ratio cycle to decorrelate.
    float frame = floor(u_time*60.0);
    float bn = hash12(uv + frame*0.137);
    float offset = fract(bn + frame*CL_GOLDEN);

    vec3 glow = vec3(0);
    vec2 b0, b1, b2;
    float distLightning = cl_distanceToScene(ro, rd, CL_MIN_DIST, CL_MAX_DIST, glow, b0, b1, b2);

    float totalT = 1.0;
    float exposure = 0.5 + 0.4*u_globalRms;
    vec3 sunDir = normalize(vec3(1.0 + 0.4*sin(u_time*0.05 + u_bass), 1.0, 1.0));
    float mu = 0.5 + 0.5*dot(rd, sunDir);
    float dCloud = 1e10;
    vec3 col = exposure * cl_mainRay(ro, rd, sunDir, totalT, mu, offset, dCloud, b0, b1, b2);
    vec3 background = cl_getSkyColour(rd);
    background += CL_SUNCOL * 0.2*cl_getGlow(1.0 - mu, 0.001, 0.55);

    float tPlane = 1e10;
    bool hitsPlane = cl_intersectPlane(vec3(0,-1,0), vec3(0, CL_CLOUD_START, 0), ro, rd, tPlane);
    if (tPlane >= dCloud) background += glow;
    col += background * totalT;
    if ((tPlane < dCloud && hitsPlane) || ro.y < CL_CLOUD_START){
        col += glow;
    }
    return col;
}

// =================================================================
// MODE 4 — DOMAIN-WARPED FBM PATTERN  (Iquilezles "warp")
// fbm(p + fbm(p + fbm(p))).  Music wiring:
//   warp depth multiplier <- u_bass
//   pattern speed         <- u_globalRms
//   color shift           <- chromaHue, drum punch flash
// =================================================================
float wf_rand(vec2 n){
    return fract(sin(dot(n, vec2(12.9898, 4.1414)))*43758.5453);
}
float wf_noise(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);
    float res = mix(mix(wf_rand(ip), wf_rand(ip+vec2(1,0)), u.x),
                    mix(wf_rand(ip+vec2(0,1)), wf_rand(ip+vec2(1,1)), u.x), u.y);
    return res*res;
}
const mat2 WF_M = mat2(0.80, 0.60, -0.60, 0.80);
float wf_fbm(vec2 p, float spd){
    float f = 0.0;
    f += 0.500000*wf_noise(p + spd); p = WF_M*p*2.02;
    f += 0.031250*wf_noise(p);       p = WF_M*p*2.01;
    f += 0.250000*wf_noise(p);       p = WF_M*p*2.03;
    f += 0.125000*wf_noise(p);       p = WF_M*p*2.01;
    f += 0.062500*wf_noise(p);       p = WF_M*p*2.04;
    f += 0.015625*wf_noise(p + sin(spd));
    return f/0.96875;
}
float wf_pattern(vec2 p, float spd, float warpAmt){
    return wf_fbm(p + warpAmt*wf_fbm(p + warpAmt*wf_fbm(p, spd), spd), spd);
}

vec3 mode_warp(vec2 uv){
    // breakdown is meant to be sparse and tense — slow tempo, low brightness,
    // chroma pushes the hue around so it still "dances".
    float spd = u_time*(0.25 + 0.5*u_globalRms);
    float warpAmt = 1.0 + 0.6*u_bass;
    vec2 p = uv * vec2(u_res.x/u_res.y, 1.0) * 2.0;
    float v = wf_pattern(p, spd, warpAmt);

    // hue from dominant chroma; saturation rides drums punch (a flash on hits)
    float hue = chromaHue();
    float sat = 0.55 + 0.35*u_drumsPunch;
    vec3 col = hsv(vec3(hue, sat, pow(v, 1.4)));
    col *= 0.45 + 0.7*u_globalRms;
    // dimming so it reads as "breakdown" rather than a hero shot
    col *= 0.7;
    return col;
}

// =================================================================
// MODE 5 — SEASCAPE  (Alexander Alekseev / TDM)
// Music wiring:
//   SEA_HEIGHT  <- u_bass
//   SEA_CHOPPY  <- u_drumsRms
//   SEA_SPEED   <- u_globalRms
//   camera ang  <- u_beatPhase
// Aesthetic darken: final color × 0.55, sky also darkened.
// =================================================================
const int   SEA_NUM_STEPS = 32;
const float SEA_EPSILON   = 1e-3;
const int   SEA_ITER_GEOMETRY = 3;
const int   SEA_ITER_FRAGMENT = 5;
const vec3  SEA_BASE = vec3(0.0, 0.07, 0.14);
const vec3  SEA_WATER_COLOR = vec3(0.6, 0.75, 0.45)*0.5;
const mat2  SEA_OCT_M = mat2(1.6, 1.2, -1.2, 1.6);

mat3 sea_fromEuler(vec3 ang){
    vec2 a1 = vec2(sin(ang.x), cos(ang.x));
    vec2 a2 = vec2(sin(ang.y), cos(ang.y));
    vec2 a3 = vec2(sin(ang.z), cos(ang.z));
    mat3 m;
    m[0] = vec3(a1.y*a3.y + a1.x*a2.x*a3.x, a1.y*a2.x*a3.x + a3.y*a1.x, -a2.y*a3.x);
    m[1] = vec3(-a2.y*a1.x, a1.y*a2.y, a2.x);
    m[2] = vec3(a3.y*a1.x*a2.x + a1.y*a3.x, a1.x*a3.x - a1.y*a3.y*a2.x, a2.y*a3.y);
    return m;
}
float sea_hash(vec2 p){
    float h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h)*43758.5453123);
}
float sea_noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return -1.0 + 2.0*mix(mix(sea_hash(i+vec2(0,0)), sea_hash(i+vec2(1,0)), u.x),
                          mix(sea_hash(i+vec2(0,1)), sea_hash(i+vec2(1,1)), u.x), u.y);
}
float sea_diff(vec3 n, vec3 l, float p){ return pow(dot(n,l)*0.4 + 0.6, p); }
float sea_spec(vec3 n, vec3 l, vec3 e, float s){
    float nrm = (s + 8.0)/(PI*8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s)*nrm;
}
vec3 sea_getSkyColor(vec3 e){
    e.y = (max(e.y, 0.0)*0.8 + 0.2)*0.8;
    return vec3(pow(1.0-e.y, 2.0), 1.0-e.y, 0.6 + (1.0-e.y)*0.4) * 0.55;
}
float sea_octave(vec2 uv, float choppy){
    uv += sea_noise(uv);
    vec2 wv = 1.0 - abs(sin(uv));
    vec2 swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x*wv.y, 0.65), choppy);
}
// SEA_TIME, SEA_HEIGHT, SEA_CHOPPY, SEA_SPEED need to be runtime-tied to music.
// Pass them through globals set by mode_seascape() before calling map().
float SEA_HEIGHT = 0.6;
float SEA_CHOPPY = 4.0;
float SEA_SPEED  = 0.8;
float SEA_FREQ   = 0.16;
float SEA_T      = 0.0;
float sea_map(vec3 p){
    float freq = SEA_FREQ, amp = SEA_HEIGHT, choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    float h = 0.0;
    for (int i = 0; i < SEA_ITER_GEOMETRY; i++){
        float d = sea_octave((uv+SEA_T)*freq, choppy);
        d += sea_octave((uv-SEA_T)*freq, choppy);
        h += d*amp;
        uv = SEA_OCT_M*uv; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}
float sea_map_d(vec3 p){
    float freq = SEA_FREQ, amp = SEA_HEIGHT, choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    float h = 0.0;
    for (int i = 0; i < SEA_ITER_FRAGMENT; i++){
        float d = sea_octave((uv+SEA_T)*freq, choppy);
        d += sea_octave((uv-SEA_T)*freq, choppy);
        h += d*amp;
        uv = SEA_OCT_M*uv; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}
vec3 sea_getNormal(vec3 p, float eps){
    vec3 n;
    n.y = sea_map_d(p);
    n.x = sea_map_d(vec3(p.x+eps, p.y, p.z)) - n.y;
    n.z = sea_map_d(vec3(p.x, p.y, p.z+eps)) - n.y;
    n.y = eps;
    return normalize(n);
}
vec3 sea_getColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist){
    float fres = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fres = min(fres*fres*fres, 0.5);
    vec3 reflected = sea_getSkyColor(reflect(eye, n));
    vec3 refracted = SEA_BASE + sea_diff(n, l, 80.0) * SEA_WATER_COLOR * 0.12;
    vec3 color = mix(refracted, reflected, fres);
    float atten = max(1.0 - dot(dist, dist)*0.001, 0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT)*0.18*atten;
    color += sea_spec(n, l, eye, 600.0*inversesqrt(dot(dist, dist)));
    return color;
}
float sea_trace(vec3 ori, vec3 dir, out vec3 p){
    float tm = 0.0, tx = 1000.0;
    float hx = sea_map(ori + dir*tx);
    if (hx > 0.0){ p = ori + dir*tx; return tx; }
    float hm = sea_map(ori);
    for (int i = 0; i < SEA_NUM_STEPS; i++){
        float tmid = mix(tm, tx, hm/(hm-hx));
        p = ori + dir*tmid;
        float hmid = sea_map(p);
        if (hmid < 0.0){ tx = tmid; hx = hmid; }
        else            { tm = tmid; hm = hmid; }
        if (abs(hmid) < SEA_EPSILON) break;
    }
    return mix(tm, tx, hm/(hm-hx));
}

vec3 mode_seascape(vec2 uv){
    // music-driven sea state
    SEA_HEIGHT = 0.45 + 0.55*u_bass;
    SEA_CHOPPY = 3.0 + 2.5*u_drumsRms;
    SEA_SPEED  = 0.45 + 0.6*u_globalRms;
    SEA_FREQ   = 0.16;
    float t = u_time * 0.3;
    SEA_T = 1.0 + t * SEA_SPEED;

    vec2 fragCoord = uv * u_res;
    vec2 nuv = fragCoord/u_res;
    nuv = nuv*2.0 - 1.0;
    nuv.x *= u_res.x/u_res.y;

    vec3 ang = vec3(sin(t*3.0)*0.1,
                    sin(t)*0.2 + 0.3 + 0.05*u_beatPhase,
                    t);
    vec3 ori = vec3(0.0, 3.5, t*5.0);
    vec3 dir = normalize(vec3(nuv.xy, -2.0));
    dir.z += length(nuv) * 0.14;
    dir = normalize(dir) * sea_fromEuler(ang);

    vec3 p;
    sea_trace(ori, dir, p);
    vec3 dist = p - ori;
    float epsN = 0.1 / u_res.x;
    vec3 n = sea_getNormal(p, dot(dist, dist)*epsN);
    vec3 light = normalize(vec3(0.0, 1.0, 0.8));
    vec3 col = mix(sea_getSkyColor(dir),
                   sea_getColor(p, n, light, dir, dist),
                   pow(smoothstep(0.0, -0.02, dir.y), 0.2));
    // donor's pow(0.65) is sRGB-ish encoding; linearize so our composite
    // pipeline doesn't double-gamma. Then darken overall + cool-tint.
    col = pow(col, vec3(0.65));
    col = pow(max(col, 0.0), vec3(2.2));
    col *= 0.45;
    col *= vec3(0.85, 0.95, 1.10);
    return col;
}

// =================================================================
// DISPATCH + NOISE-DISSOLVE TRANSITION
// =================================================================
vec3 dispatch(int mode, vec2 uv){
    if (mode == 0) return mode_intro(uv);
    if (mode == 1) return mode_protean(uv);
    if (mode == 2) return mode_drop(uv);
    if (mode == 3) return mode_clouds(uv);
    if (mode == 4) return mode_warp(uv);
    return mode_seascape(uv);
}

void main(){
    vec3 col = dispatch(u_modeA, v_uv);
    if (u_modeBlend > 0.001){
        vec3 b = dispatch(u_modeB, v_uv);

        // Noise-dissolve: a slow-moving warped FBM acts as a per-pixel
        // threshold so the new mode breaks through in organic blobs rather
        // than a uniform fade. A bright edge glow on the dissolve front
        // makes the cut feel like a wipe.
        vec2 dp = v_uv * vec2(u_res.x/u_res.y, 1.0);
        float n = warpedFbm(dp*3.5 + vec2(0.0, u_time*0.15), u_time*0.5);
        float blend = u_modeBlend;
        float edge = 0.10;
        float t = smoothstep(blend - edge, blend + edge, n);
        col = mix(col, b, 1.0 - t);

        // glow band at the dissolve front, brightest mid-transition
        float front = exp(-pow((n - blend)/edge*1.4, 2.0));
        float pulse = sin(blend*PI);   // 0 at start/end, 1 in middle
        col += vec3(0.45, 0.65, 1.0) * front * pulse * 0.55;
    }

    // gentle global beat-phase brightness lift
    col *= 1.0 + 0.05 * (1.0 - u_beatPhase);

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
    float w = smoothstep(u_threshold, u_threshold + 0.25, b);
    frag = vec4(c * w * w, 1.0);
}
"""

BLUR_FS = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex;
uniform vec2 u_dir;
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
float hash12g(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main(){
    vec2 dir = v_uv - 0.5;
    float r2 = dot(dir, dir);

    vec2 ofs = dir * u_caAmt;
    vec3 scene;
    scene.r = texture(u_scene, v_uv + ofs).r;
    scene.g = texture(u_scene, v_uv).g;
    scene.b = texture(u_scene, v_uv - ofs).b;

    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 col = scene + bloom * u_bloomAmt;
    col *= u_exposure;
    col = aces(col);

    col *= 1.0 - r2 * u_vignette;

    // triangular-PDF grain with sin-free hash (no diagonal banding)
    vec2 gp = v_uv * vec2(1920.0, 1080.0);
    float r1  = hash12g(gp + u_time*60.0);
    float r2g = hash12g(gp + u_time*60.0 + 113.7);
    float g = (r1 + r2g) - 1.0;
    col += vec3(g) * u_grainAmt;

    col = pow(max(col, 0.0), vec3(1.0/2.2));
    frag = vec4(col, 1.0);
}
"""
