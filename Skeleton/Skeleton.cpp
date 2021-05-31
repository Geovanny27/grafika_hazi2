//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tierra Coral Luis Geovanny
// Neptun : FGB2MU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks;
	float shininess=0;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};
struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) :Material(ROUGH) {
		ka = _kd * (float)M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};
vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial :Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) :Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
class Intersectable {
protected:
	Material* material;
	bool shadow = true;
public:
	virtual Hit intersect(const Ray& ray) = 0;
	bool Shadow() {
		return shadow;
	}
};
class Cylinder :public Intersectable {
	vec3 bottom;
	float radius, height;
public:
	Cylinder(const vec3& _bottom, float _radius, float _height, Material* _material) { bottom = _bottom; radius = _radius; height = _height; material = _material; }
	bool outOfLimits(float y) {
		if (y<bottom.y || y>bottom.y + height)return true;
		return false;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec2 dir = vec2(ray.dir.x, ray.dir.z);
		vec2 start = vec2(ray.start.x - bottom.x, ray.start.z - bottom.z);
		float a = dot(dir, dir) / radius / radius;
		float b = 2 * dot(dir, start) / radius / radius;
		float c = dot(start, start) / radius / radius - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0)return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a; //a nagyobb
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)return hit;
		float t;
		vec3 pos;
		if (t2 > 0) {
			t = t2;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y)) {
				t = t1;
				pos = ray.start + ray.dir * t;
				if (outOfLimits(pos.y))return hit;
			}
		}
		else {
			t = t1;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y))return hit;
		}
		hit.t = t;
		hit.position = pos;
		hit.normal = normalize(vec3(pos.x - bottom.x, 0, pos.z - bottom.z));
		hit.material = material;
		return hit;
	}

};
class Paraboloid :public Intersectable {
	vec3 top;
	float A, B, height;
public:
	Paraboloid(const vec3& _top, float _A, float _B, float _height, Material* _material) { top = _top; A = _A; B = _B; height = _height; material = _material; }
	bool outOfLimits(float y) {
		if (y > top.y || y < top.y - height)return true;
		return false;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = vec3(ray.start.x - top.x, ray.start.y - top.y, ray.start.z - top.z);
		float a = ray.dir.x * ray.dir.x / A / A + (ray.dir.z * ray.dir.z / B / B);
		float b = 2 * start.x * ray.dir.x / A / A + (2 * start.z * ray.dir.z / B / B) + ray.dir.y;
		float c = start.x * start.x / A / A + (start.z * start.z / B / B) + start.y;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0)return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)return hit;
		float t;
		vec3 pos;
		if (t2 > 0) {
			t = t2;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y)) {
				t = t1;
				pos = ray.start + ray.dir * t;
				if (outOfLimits(pos.y))return hit;
			}
		}
		else {
			t = t1;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y))return hit;
		}
		hit.t = t;
		hit.position = pos;
		pos = start + ray.dir * t;
		hit.normal = normalize(vec3(-2 * pos.x / A / A, -1, -2 * pos.z / B / B));
		hit.material = material;
		return hit;
	}
};
class Hyperboloid :public Intersectable {
	vec3 center;
	float A, B, C, height, ratio;
public:
	Hyperboloid(const vec3& _bottom, float _A, float _B, float _C, float _height, float _ratio, Material* _material, bool _shadow = true) { center = _bottom; A = _A; B = _B; C = _C; height = _height; ratio = _ratio; material = _material; shadow = _shadow; }
	bool outOfLimits(float y) {
		if (y<center.y - (1 - ratio) * height || y>center.y + height * ratio)return true;
		return false;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = vec3(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z);
		float a = ray.dir.x * ray.dir.x / A / A + (ray.dir.z * ray.dir.z / B / B) - (ray.dir.y * ray.dir.y / C / C);
		float b = 2 * start.x * ray.dir.x / A / A + (2 * start.z * ray.dir.z / B / B) - (2 * start.y * ray.dir.y / C / C);
		float c = start.x * start.x / A / A + (start.z * start.z / B / B) - (start.y * start.y / C / C) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0)return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)return hit;
		float t;
		vec3 pos;
		vec3 pos1=ray.start+ray.dir*t1;
		vec3 pos2=ray.start+ray.dir*t2;
		if (t2 > 0) {
			pos = ray.start + ray.dir * t2;
			t = t2;
			if (outOfLimits(pos.y)) {
				pos = ray.start + ray.dir * t1;
				if (outOfLimits(pos.y))return hit;
				t = t1;
			}
		}
		else {
			t = t1;
			pos= ray.start + ray.dir * t;
			if (outOfLimits(pos.y))return hit;
		}
		hit.t = t;
		hit.position = pos;
		pos = start + ray.dir * t;
		hit.normal= normalize(vec3(2 * pos.x / A / A, -2 * pos.y / C / C, 2 * pos.z / B / B));
		hit.material = material;
		return hit;
	}
};
class Ellipsoid :public Intersectable {
	vec3 center;
	float A, B, C;
public:
	Ellipsoid(const vec3& _center, float _A, float _B, float _C, Material* _material) { center = _center; A = _A; B = _B; C = _C; material = _material; }
	bool outOfLimits(float y) {
		if (y > center.y + 0.95f)return true;
		return false;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = vec3(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z);
		float a = ray.dir.x * ray.dir.x / A / A + (ray.dir.z * ray.dir.z / B / B) + (ray.dir.y * ray.dir.y / C / C);
		float b = 2 * start.x * ray.dir.x / A / A + (2 * start.z * ray.dir.z / B / B) + (2 * start.y * ray.dir.y / C / C);
		float c = start.x * start.x / A / A + (start.z * start.z / B / B) + (start.y * start.y / C / C) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0)return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)return hit;
		float t;
		vec3 pos;
		if (t2 > 0) {
			t = t2;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y)) {
				t = t1;
				pos = ray.start + ray.dir * t;
				if (outOfLimits(pos.y))return hit;
			}
		}
		else {
			t = t1;
			pos = ray.start + ray.dir * t;
			if (outOfLimits(pos.y))return hit;
		}
		hit.t = t;
		hit.position = pos;
		pos = start + ray.dir * hit.t;
		hit.normal = normalize(vec3(2 * pos.x / A / A, 2 * pos.y / C / C, 2 * pos.z / B / B));
		hit.material = material;
		return hit;
	}
};
class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat, fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;
float random2() { return (float)rand() / RAND_MAX; }
class Scene {
	std::vector<Intersectable*>objects;
	std::vector<Light*>lights;
	Camera camera;
	vec3 La;
	int n = 15;
	vec3* samples;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 70 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(0.0f, 1.0f, 0.4f), Le(20.0f, 20.0f, 20.0f);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f),kd3(0.0f,0.4f,0.4f), ks(2.0f, 2.0f, 2.0f);
		Material* material1 = new RoughMaterial(kd1, ks, 50);
		Material* material2 = new RoughMaterial(kd2, ks, 50);
		Material* green = new RoughMaterial(kd3, ks, 50);
		Material* gold = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		Material* silver = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		objects.push_back(new Ellipsoid(vec3(0.0f, 0.0f, 0.0f), 2.0f, 2.0f, 1.0f, material1));
		objects.push_back(new Hyperboloid(vec3(0.0f, 0.95f, 0.0f), sqrtf(0.39f), sqrtf(0.39f), 1.2f, 1.2f, 1.0f, silver,false));
		objects.push_back(new Cylinder(vec3(-0.9f, -1.0f, -0.6f), 0.3f, 1.2f, green));
		objects.push_back(new Paraboloid(vec3(0.7f, -0.2f, -0.4f), 0.5f, 0.7f, 0.8f, gold));
		objects.push_back(new Hyperboloid(vec3(-0.1f, -0.45f, -1.2f), 0.1f, 0.1f, 0.1f, 1.0f, 0.5f, material2));

	}
	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		samples = new vec3[n];
		Samples(samples);

		for (int Y = 0; Y < windowHeight; Y++)
		{
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++)
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
		printf("Rendering time: %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
		delete[] samples;
	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects)
		{
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) { 
			if (object->Shadow()) {
				if(object->intersect(ray).t > 0)return true;
			}
		}
		return false;
	}

	void Samples(vec3 samples[]) {
		for (int i = 0; i < n; i++)
		{
			samples[i] = vec3(random2() * sqrtf(0.39f), 0.95f, random2() * sqrtf(0.39f));
		}
	}
	vec3 Fresnel(Hit hit, Ray ray) {
		float cosa = -dot(ray.dir, hit.normal);
		vec3 one(1, 1, 1);
		return hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
	}
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5)return La;
		Hit hit = firstIntersect(ray);
		vec3 sky(0.5294f, 0.8078f, 0.9215f);
		if (hit.t < 0)return sky + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);

		vec3 outRadience(0, 0, 0);
		if (hit.material->type == ROUGH) {
			outRadience = hit.material->ka * La;
			for (int i = 0; i < n; i++)
			{
				vec3 inDir = normalize(samples[i]-hit.position);
				Ray shadowRay(hit.position + hit.normal * epsilon, inDir);
				float cosThetaIn = dot(hit.normal, inDir);
				float cosTheta = -dot(vec3(0, -1, 0), inDir);
				float deltaOmega = 0.39f * (float)M_PI / n * cosTheta / powf(length(hit.position - samples[i]), 2);
				if (cosThetaIn > 0 && !shadowIntersect(shadowRay)){
					vec3 inRad = trace(Ray(hit.position + epsilon * hit.normal, inDir), depth + 1);
					outRadience = outRadience + inRad * hit.material->kd * cosThetaIn * deltaOmega;
					vec3 halfway = normalize(-ray.dir + inDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadience = outRadience + inRad * hit.material->ks * powf(cosDelta, hit.material->shininess) * deltaOmega;
				}
			}

		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			vec3 F = Fresnel(hit, ray);
			outRadience = outRadience + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadience;
	}
};

GPUProgram gpuProgram;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 cVertexPosition;	//attrib array 0
	out vec2 texcoord;

	void main() {
		texcoord=(cVertexPosition+vec2(1,1))/2;
		gl_Position=vec4(cVertexPosition.x,cVertexPosition.y,0,1);
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 texcoord;		// uniform variable, the color of the primitive
	out vec4 fragmentColor;		// computed color of the current pixel

	void main() {
		fragmentColor = texture(textureUnit, texcoord);	// computed color is the color of the primitive
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1,-1,1,-1,1,1,-1,1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}
	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};
Scene scene;
FullScreenTexturedQuad* fullScreenTextureQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTextureQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}


void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTextureQuad->LoadTexture(image);
	fullScreenTextureQuad->Draw();
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {}


void onKeyboardUp(unsigned char key, int pX, int pY) {}


void onMouseMotion(int pX, int pY) {}


void onMouse(int button, int state, int pX, int pY) {}


void onIdle() {
}
