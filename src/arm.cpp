#include <FL/Enumerations.H>
#include <FL/Fl.H>
#include <FL/Fl_Widget.H>
#include <FL/Fl_Window.H>
#include <FL/fl_draw.H>
#include <chrono>
#include <thread>

#include "Common.h"
#include "MathUtils.h"
#include "Vec.h"

using namespace Math;

constexpr int WIDTH = 1500;
constexpr int HEIGHT = 1500;
constexpr int FPS = 60;
constexpr double SCALE = 150;

constexpr double ANGULAR_SPEED = PI / 6;

struct Geometry : public Fl_Widget {
  virtual void draw() = 0;
  Geometry(const Geometry &other) : Fl_Widget(0, 0, WIDTH, HEIGHT) {}

  Geometry() : Fl_Widget(0, 0, WIDTH, HEIGHT) {}

protected:
  double scale(double x) { return x * SCALE; }
};

struct Circle : public Geometry {
  Point2d pos;
  double radius;
  Fl_Color fill;

  Circle(const Point2d &pos, double radius, Fl_Color color = FL_BLACK)
      : Geometry(), pos(pos), radius(radius), fill(color) {}

  void draw() override {
    fl_color(fill);
    int x = scale(pos.x() - radius);
    int y = scale(pos.y() - radius);
    int dia = scale(2 * radius);
    fl_pie(x, y, dia, dia, 0, 360);
  }
};

struct Target : public Circle {
  Target(const Point2d &pos, double radius) : Circle(pos, radius, FL_BLACK) {}

  int handle(int event) override {
    switch (event) {
    case FL_PUSH:
      pos[0] = Fl::event_x() / SCALE;
      pos[1] = Fl::event_y() / SCALE;

      redraw();
      return 1;
    default:
      return 0;
    }
  }
};

struct Bone : public Geometry {
  Point2d upper, lower;
  double width, length;
  Fl_Color fill;

  Bone(const Point2d &lower, const Point2d &upper, Fl_Color color = FL_BLACK)
      : fill(color), upper(upper), lower(lower) {
    length = (upper - lower).length();
    width = length / 10;
  }

  void draw() override {
    fl_color(fill);
    Vec2d norm =
        Vec2d({lower[1] - upper[1], upper[0] - lower[0]}).to_length(width);
    fl_begin_polygon();
    fl_color(fill);
    fl_vertex(scale((upper - norm).x()), scale((upper - norm).y()));
    fl_vertex(scale((upper + norm).x()), scale((upper + norm).y()));
    fl_vertex(scale((lower + norm).x()), scale((lower + norm).y()));
    fl_vertex(scale((lower - norm).x()), scale((lower - norm).y()));
    fl_end_polygon();
  }
};

struct Limb : public Geometry {
  vector<Bone> joints;

  Limb(const vector<Bone> &joints) : joints(joints) {}

  static Limb from_vector(const Point2d &origin,
                          const vector<std::pair<double, Fl_Color>> &arr) {
    vector<Bone> joints;
    Point2d last = origin;
    for (int i = 0; i < arr.size(); ++i) {
      Vec2d upper{last[0], last[1] + arr[i].first};
      joints.emplace_back(last, upper, arr[i].second);
      last = upper;
    }
    return Limb(joints);
  }

  double solve(const Point2d &goal, double dt) {
    Point2d &end = end_pos();
    for (int i = joints.size() - 1; i >= 0; --i) {
      Bone &joint = joints[i];
      Vec2d toEnd = (end - joint.lower).normalized();
      Vec2d toGoal = (goal - joint.lower).normalized();

      double a = acos(clamp(toEnd.dot(toGoal), -1, 1));
      a = a * dt;
      a = min(a, ANGULAR_SPEED * dt);

      if (toEnd.cross(toGoal) < 0) {
        a = -a;
      }

      rotate_bone(joint, a);

      update_positions(i);
    }
    return (end - goal).length();
  }

  void draw() override {
    for (auto &joint : joints) {
      joint.draw();
    }
  }

private:
  Point2d &end_pos() { return joints.back().upper; }

  void rotate_bone(Bone &bone, double a) {
    Vec2d boneVec = bone.upper - bone.lower;
    Vec2d rotatedVec{boneVec.x() * cos(a) - boneVec.y() * sin(a),
                     boneVec.x() * sin(a) + boneVec.y() * cos(a)};
    bone.upper = bone.lower + rotatedVec.to_length(boneVec.length());
  }

  void update_positions(int fromJoint) {
    for (int i = fromJoint; i < joints.size() - 1; ++i) {
      Vec2d diff = joints[i].upper - joints[i + 1].lower;
      joints[i + 1].lower += diff;
      joints[i + 1].upper += diff;
    }
  }
};

struct Body : public Geometry {
  Limb leftArm, RightArm;
  Circle head;
  Circle eyeLeft, eyeRight;

  Body(const Limb &leftArm, const Limb &rightArm, const Circle &head,
       const Circle &eyeLeft, const Circle &eyeRight)
      : leftArm(leftArm), RightArm(rightArm), head(head), eyeLeft(eyeLeft),
        eyeRight(eyeRight) {}

  void draw() override {
    leftArm.draw();
    RightArm.draw();
    head.draw();
    eyeLeft.draw();
    eyeRight.draw();
  }
};

class Canvas : public Fl_Window {
public:
  Canvas(int W, int H, const char *l = 0) : Fl_Window(W, H, l) {}

  void draw() override {
    fl_color(FL_BACKGROUND_COLOR);
    fl_rectf(0, 0, w(), h());
    Fl_Window::draw();
  }
};

Canvas *canvas = new Canvas(WIDTH, HEIGHT, "Project3");
Target target({5, 9.5}, 0.1);

Body body{Limb::from_vector({3, 2.5},
                            {
                                {2.0, FL_DARK_RED},
                                {1.5, FL_RED},
                                {0.5, FL_DARK_MAGENTA},
                                {0.2, FL_DARK_BLUE},
                            }),
          Limb::from_vector({7, 2.5},
                            {
                                {2.0, FL_DARK_RED},
                                {1.5, FL_RED},
                                {0.5, FL_DARK_MAGENTA},
                                {0.2, FL_DARK_BLUE},
                            }),
          {{5, 2}, 0.5, FL_GREEN},
          {{4.7, 2}, 0.05, FL_BLACK},
          {{5.3, 2}, 0.05, FL_BLACK}};

void update(void *) {
  canvas->redraw();
  body.leftArm.solve(target.pos, 0.1);
  body.RightArm.solve(target.pos, 0.1);
  std::this_thread::sleep_for(
      std::chrono::milliseconds((int)(1.0 / FPS * 1000.0)));
}

int main() {
  canvas->end();
  canvas->show();
  Fl::add_idle(update);

  return Fl::run();
}
