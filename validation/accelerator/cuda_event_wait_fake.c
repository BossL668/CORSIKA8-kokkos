/* CPU-only fake libcudart and its driver; NEVER loads a real CUDA library.
 * Compile with C8_FAKE_LIBRARY for the shared library, otherwise for the driver.
 */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct CUevent_st { int live; unsigned int flags; };
typedef struct CUevent_st *Event;
typedef struct CUstream_st *Stream;

#ifdef C8_FAKE_LIBRARY
static _Thread_local struct CUevent_st slots[16];
static _Thread_local int last_error, input_errno, last_error_reads;
static _Thread_local unsigned int last_flags;
static _Thread_local uintptr_t last_stream;
static _Thread_local char trace[256];
static _Thread_local size_t trace_size;
static int finish(char api, int result) {
  input_errno = errno;
  assert(trace_size + 2 < sizeof(trace));
  trace[trace_size++] = api;
  trace[trace_size] = 0;
  last_error = result + 1000;
  errno = 2000 + result;
  return result;
}
int fakeInputErrno(void) { return input_errno; }
int fakePeekError(void) { return last_error; }
int fakeErrorReads(void) { return last_error_reads; }
const char *fakeTrace(void) { return trace; }
unsigned int fakeFlags(void) { return last_flags; }
uintptr_t fakeStream(void) { return last_stream; }
int cudaGetLastError(void) {
  ++last_error_reads;
  int result = last_error;
  last_error = 0;
  return result;
}
static int valid(Event event) {
  for (size_t i=0; i<16; ++i) if (event == &slots[i]) return slots[i].live;
  return 0;
}
int cudaEventCreateWithFlags(Event *event, unsigned int flags) {
  last_flags = flags;
  if (!event || flags == 99) return finish('C', 17);
  for (size_t i=0; i<16; ++i) if (!slots[i].live) {
    slots[i].live=1; slots[i].flags=flags; *event=&slots[i];
    return finish('C', 0);
  }
  return finish('C', 2);
}
int cudaEventRecord(Event event, Stream stream) {
  last_stream = (uintptr_t)stream;
  return finish('R', valid(event) ? 0 : 400);
}
#ifndef C8_FAKE_OMIT_SYNC
int cudaEventSynchronize(Event event) {
  int saved_errno = errno;
  if (!valid(event)) return finish('S', 400);
  /* Tiny wait only, never a physics workload or a GPU call. */
  struct timespec pause = {0, 1000000};
  nanosleep(&pause, NULL);
  errno = saved_errno;
  return finish('S', 0);
}
#endif
int cudaEventDestroy(Event event) {
  if (!valid(event)) return finish('D', 400);
  if (event->flags == 13) return finish('D', 41);
  event->live = 0;
  return finish('D', 0);
}
#else
int cudaEventCreateWithFlags(Event *, unsigned int);
int cudaEventRecord(Event, Stream);
int cudaEventSynchronize(Event);
int cudaEventDestroy(Event);
int cudaGetLastError(void);
int fakeInputErrno(void);
int fakePeekError(void);
int fakeErrorReads(void);
const char *fakeTrace(void);
unsigned int fakeFlags(void);
uintptr_t fakeStream(void);

#define CHECK(call, expected) do { \
  errno = 321; \
  int result = (call); \
  assert(result == (expected)); \
  assert(errno == 2000 + (expected)); \
  assert(fakeInputErrno() == 321); \
  assert(fakePeekError() == 1000 + (expected)); \
  assert(fakeErrorReads() == 0); \
} while (0)

static void lifecycle(void) {
  Event event = NULL;
  CHECK(cudaEventCreateWithFlags(&event, 1), 0);
  assert(fakeFlags() == 1);
  Event first = event;
  CHECK(cudaEventRecord(event, (Stream)(uintptr_t)0x12340000), 0);
  assert(fakeStream() == (uintptr_t)0x12340000);
  CHECK(cudaEventSynchronize(event), 0);
  CHECK(cudaEventDestroy(event), 0);
  CHECK(cudaEventSynchronize(event), 400);
  CHECK(cudaEventCreateWithFlags(&event, 0), 0);
  assert(event == first && fakeFlags() == 0); /* Reused address, different flags. */
  CHECK(cudaEventCreateWithFlags(&event, 99), 17);
  assert(event == first && fakeFlags() == 99); /* Failure must not change map. */
  CHECK(cudaEventRecord(event, NULL), 0);
  assert(fakeStream() == 0);
  CHECK(cudaEventSynchronize(event), 0);
  CHECK(cudaEventSynchronize(event), 0);
  CHECK(cudaEventDestroy(event), 0);
  CHECK(cudaEventCreateWithFlags(&event, 13), 0);
  CHECK(cudaEventDestroy(event), 41); /* Failed destroy must retain flags. */
  CHECK(cudaEventSynchronize(event), 0);
  assert(!strcmp(fakeTrace(), "CRSDSCCRSSDCDS"));
  assert(cudaGetLastError() == 1000 && fakePeekError() == 0 && fakeErrorReads() == 1);
  puts("FAKE_DRIVER_LIFECYCLE_PASS");
}

static pthread_barrier_t barrier;
static void *thread_work(void *argument) {
  unsigned int flags = (unsigned int)(uintptr_t)argument;
  pthread_barrier_wait(&barrier);
  for (int i=0; i<3; ++i) {
    Event event;
    CHECK(cudaEventCreateWithFlags(&event, flags), 0);
    CHECK(cudaEventRecord(event, (Stream)(uintptr_t)(flags+1)), 0);
    CHECK(cudaEventSynchronize(event), 0);
    CHECK(cudaEventDestroy(event), 0);
  }
  assert(!strcmp(fakeTrace(), "CRSDCRSDCRSD"));
  return NULL;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "lifecycle")) lifecycle();
  else if (!strcmp(argv[1], "threads")) {
    pthread_t a,b;
    assert(!pthread_barrier_init(&barrier,NULL,2));
    assert(!pthread_create(&a,NULL,thread_work,(void *)(uintptr_t)0));
    assert(!pthread_create(&b,NULL,thread_work,(void *)(uintptr_t)1));
    assert(!pthread_join(a,NULL) && !pthread_join(b,NULL));
    assert(!pthread_barrier_destroy(&barrier));
    puts("FAKE_DRIVER_THREADS_PASS");
  } else if (!strcmp(argv[1], "capacity")) {
    Event events[4];
    for (int i=0; i<4; ++i) CHECK(cudaEventCreateWithFlags(&events[i],(unsigned)i),0);
    for (int i=0; i<4; ++i) CHECK(cudaEventRecord(events[i],NULL),0);
    for (int i=0; i<4; ++i) CHECK(cudaEventDestroy(events[i]),0);
    puts("FAKE_DRIVER_CAPACITY_PASS");
  } else if (!strcmp(argv[1], "quiet")) {
    puts("FAKE_DRIVER_NO_CUDA_CALLS");
  } else return 2;
  return 0;
}
#endif
