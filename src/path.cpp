#include <FL/Enumerations.H>
#include <FL/Fl.H>
#include <FL/Fl_Widget.H>
#include <FL/Fl_Window.H>
#include <FL/fl_draw.H>
#include <chrono>
#include <queue>
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
constexpr double LINEAR_SPEED = 0.1;
constexpr int NUM_OBSTACLES = 15;
constexpr int NUM_NODES = 300;

int pathIndex = -1;

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
      pathIndex = -1;
      cout << "target set" << endl;

      redraw();
      return 1;
    default:
      return 0;
    }
  }
};

struct Agent : public Geometry {
  Point2d pos;
  Vec2d vel;

  Agent(const Point2d &pos) : pos(pos) {
    vel = Vec2d{{1, 1}}.normalized() * LINEAR_SPEED;
  }

  void draw() override {
    fl_color(FL_BLUE);
    Vec2d n = vel.normalized();
    Vec2d n1 = Vec2d{-n.y(), n.x()}.normalized();
    Point2d ll = pos - n * 0.2 - n1 * 0.1;
    Point2d lr = pos - n * 0.2 + n1 * 0.1;
    Point2d ur = pos + n * 0.2 + n1 * 0.1;
    Point2d ul = pos + n * 0.2 - n1 * 0.1;
    fl_begin_polygon();
    fl_color(FL_BLUE);
    fl_vertex(scale(ll.x()), scale(ll.y()));
    fl_vertex(scale(lr.x()), scale(lr.y()));
    fl_vertex(scale(ur.x()), scale(ur.y()));
    fl_vertex(scale(ul.x()), scale(ul.y()));
    fl_end_polygon();
  }
};

struct Node : Geometry {
  Point2d pos;
  vector<Node *> neighbors;

  Node(const Point2d &p) : pos(p) {}

  double distance(const Node &other) const {
    return (pos - other.pos).length();
  }

  void draw() override {
    fl_color(FL_BLACK);
    int x = scale(pos.x() - 0.01);
    int y = scale(pos.y() - 0.01);
    int dia = scale(2 * 0.01);
    fl_pie(x, y, dia, dia, 0, 360);
    for (auto neighbor : neighbors) {
      fl_line(scale(pos.x()), scale(pos.y()), scale(neighbor->pos.x()),
              scale(neighbor->pos.y()));
    }
  }
};

class Graph : Geometry {
public:
  std::vector<Node> nodes;

  void add_node(const Point2d &point) { nodes.emplace_back(point); }

  void add_edge(int node1, int node2) {
    nodes[node1].neighbors.push_back(&nodes[node2]);
    nodes[node2].neighbors.push_back(&nodes[node1]);
  }

  void draw() override {
    for (auto &node : nodes) {
      node.draw();
    }
  }

  int closest_node(const Point2d &point) {
    int closest = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nodes.size(); i++) {
      double dist = nodes[i].distance({point});
      if (dist < min_dist) {
        min_dist = dist;
        closest = i;
      }
    }
    return closest;
  }

  vector<Node *> AStar(const Point2d &agent, const Point2d &target) {
    Node *start = &nodes[closest_node(agent)];
    Node *goal = &nodes[closest_node(target)];

    std::unordered_map<Node *, Node *> cameFrom;
    std::unordered_map<Node *, double> gScore;
    for (auto &node : nodes) {
      gScore[&node] = std::numeric_limits<double>::infinity();
    }
    gScore[start] = 0;

    auto heuristic = [goal](Node *node) { return node->distance(*goal); };

    std::priority_queue<std::pair<double, Node *>,
                        vector<std::pair<double, Node *>>, std::greater<>>
        frontier;
    frontier.emplace(0, start);

    while (!frontier.empty()) {
      Node *current = frontier.top().second;
      frontier.pop();

      if (current == goal) {
        vector<Node *> path;
        while (current != start) {
          path.push_back(current);
          current = cameFrom[current];
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
      }

      for (Node *neighbor : current->neighbors) {
        double tentative_gScore =
            gScore[current] + current->distance(*neighbor);
        if (tentative_gScore < gScore[neighbor]) {
          cameFrom[neighbor] = current;
          gScore[neighbor] = tentative_gScore;
          double fScore = tentative_gScore + heuristic(neighbor);
          frontier.emplace(fScore, neighbor);
        }
      }
    }

    return {};
  }

  vector<Node *> bfs(const Point2d &agent, const Point2d &target) {
    Node *start = &nodes[closest_node(agent)];
    Node *goal = &nodes[closest_node(target)];

    std::queue<Node *> queue;
    std::unordered_map<Node *, Node *> cameFrom;
    std::unordered_map<Node *, bool> visited;

    queue.push(start);
    visited[start] = true;

    while (!queue.empty()) {
      Node *current = queue.front();
      queue.pop();

      if (current == goal) {
        std::vector<Node *> path;
        while (current != start) {
          path.push_back(current);
          current = cameFrom[current];
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
      }

      for (Node *neighbor : current->neighbors) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          cameFrom[neighbor] = current;
          queue.push(neighbor);
        }
      }
    }

    return {};
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
vector<Node *> path;

vector<Circle> obstacles{};
Graph graph{};
Agent agent{{1, 1}};
Target target{{5, 5}, 0.1};

void init_obstacles() {
  for (int i = 0; i < NUM_OBSTACLES; i++) {
    Point2d pos{random_double(0, 10), random_double(0, 10)};
    double radius = random_double(0.1, 0.5);
    obstacles.emplace_back(pos, radius, FL_RED);
  }
}

bool line_circle_collision(const Point2d &p1, const Point2d &p2,
                           const Circle &c) {
  Vec2d lineVec = p2 - p1;
  Vec2d circleVec = c.pos - p1;
  double t = circleVec.dot(lineVec) / lineVec.dot(lineVec);
  t = std::max(0.0, std::min(t, 1.0));
  Vec2d closestPoint = p1 + t * lineVec;
  return (closestPoint - c.pos).length() <= c.radius;
}

void init_graph() {
  for (int i = 0; i < NUM_NODES; i++) {
    Point2d pos{random_double(0, 10), random_double(0, 10)};
    graph.add_node(pos);
  }
  for (int i = 0; i < NUM_NODES; i++) {
    for (int j = 0; j < NUM_NODES; j++) {
      Point2d pos1 = graph.nodes[i].pos;
      Point2d pos2 = graph.nodes[j].pos;
      if ((pos1 - pos2).length() > 2) {
        continue;
      }
      if (std::any_of(obstacles.begin(), obstacles.end(),
                      [pos1, pos2](const Circle &c) {
                        return line_circle_collision(pos1, pos2, c);
                      })) {
        continue;
      }
      graph.add_edge(i, j);
    }
  }
}

void move_towards_next_node(Agent &agent, double dt) {
  if (pathIndex == -1 || pathIndex >= path.size()) {
    return;
  }
  Node *node = path[pathIndex];
  Vec2d diff = node->pos - agent.pos;
  if (diff.length() < LINEAR_SPEED * dt) {
    pathIndex++;
    return;
  }
  agent.vel = diff.normalized() * LINEAR_SPEED;
  agent.pos += agent.vel * dt;
}

void update(void *) {
  if (pathIndex == -1) {
    path = graph.bfs(agent.pos, target.pos);
    if (!path.empty()) {
      pathIndex = 0;
    }
    cout << "path found" << endl;
  }

  move_towards_next_node(agent, 0.1);

  canvas->redraw();
  std::this_thread::sleep_for(
      std::chrono::milliseconds((int)(1.0 / FPS * 1000.0)));
}

int main() {
  init_obstacles();
  init_graph();
  canvas->end();
  canvas->show();
  Fl::add_idle(update);

  return Fl::run();
}
