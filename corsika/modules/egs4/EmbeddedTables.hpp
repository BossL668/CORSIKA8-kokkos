#pragma once
#include <memory>
namespace c7_egs4::application {
class Session;
std::unique_ptr<Session> makeEmbeddedSession();
char const* embeddedTableSha256();
}
