#include <corsika/accelerator/em/detail/CooperativeScheduleJournal.hpp>
#include <iostream>
#include <sstream>
using namespace corsika::accelerator::em::detail;
void require(bool b) { if (!b) throw std::runtime_error("journal test failed"); }
template<class F> void rejects(F&& f, std::string const& expected) {
  try { f(); } catch (std::exception const& e) {
    require(std::string(e.what()).find(expected) != std::string::npos);return;
  }
  throw std::runtime_error("journal accepted invalid state");
}
int main() {
  auto const hash = std::string(64, 'a');
  CooperativeScheduleRecord record{CooperativeAction::Dispatch, CooperativeEndpoint::Cuda, 1, 256, {1, 513}, hash};
  std::ostringstream output;
  CooperativeScheduleWriter writer(output, hash);
  writer.append(record); record.action=CooperativeAction::Commit; writer.append(record); writer.finish();
  rejects([&]{writer.finish();}, "already closed");
  rejects([&]{writer.append(record);}, "already closed");
  std::istringstream input(output.str()); CooperativeScheduleReader reader(input,hash);
  auto actual=record;actual.action=CooperativeAction::Dispatch;
  reader.expect(actual);reader.expect(record);reader.finish();
  rejects([&]{reader.next();},"already finished");
  std::istringstream changed(output.str()); CooperativeScheduleReader mismatch(changed,hash);
  actual.endpoint=CooperativeEndpoint::OpenMP;
  rejects([&]{mismatch.expect(actual);}, "record 0: endpoint");
  rejects([&]{mismatch.next();}, "previously failed");
  std::istringstream wrong(output.str());
  rejects([&]{CooperativeScheduleReader r(wrong,std::string(64,'b'));}, "configuration mismatch");
  std::istringstream truncated(output.str().substr(0,output.str().find("END")));
  CooperativeScheduleReader cut(truncated,hash);cut.next();cut.next();
  rejects([&]{cut.finish();},"truncated");
  std::istringstream oversized("C8COOP 1 "+hash+"\n"+std::string(2048,'x')+"\n");
  CooperativeScheduleReader huge(oversized,hash);
  rejects([&]{huge.next();},"oversized");
  std::ostringstream broken;broken.setstate(std::ios::badbit);
  rejects([&]{CooperativeScheduleWriter w(broken,hash);},"write failed");
  std::cout << "PASS: streaming journal, exact replay, first divergence, truncation and size guards\n";
}
