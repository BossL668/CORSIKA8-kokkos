/* Validation-only Linux libcudart interposer. No CUDA headers/runtime linkage.
 *
 * cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror \
 *    cuda_event_wait_preload.c -o libcuda_event_wait_timer.so -ldl -pthread
 *
 * CUDA 12's public C ABI: cudaError_t is an int, and event/stream are opaque
 * pointers. This observes the ordinary APIs, not *_ptsz or driver cuEvent APIs.
 * No extra CUDA call is made, including cudaGetLastError. Destroy is intercepted
 * ONLY to maintain create/reuse lifetime information. All original calls and
 * arguments are forwarded once, with their return and errno preserved.
 *
 * The API timer excludes bookkeeping, locks, and log output. These still perturb
 * the process: this is an opt-in wait diagnosis, NEVER an uninstrumented speed
 * measurement. Wall duration is host API duration, not GPU kernel time. A CUDA
 * blocking event can consume CPU internally despite cudaEventBlockingSync.
 *
 * Bounded static tables; exit-time JSON-lines on stderr, no output file. A killed
 * process may not print summaries. Missing symbols cause an explicit exit 127.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef C8_EVENT_TIMER_CAPACITY
#define C8_EVENT_TIMER_CAPACITY 8192
#endif
#ifndef C8_EVENT_TIMER_ROWS
#define C8_EVENT_TIMER_ROWS 4096
#endif

typedef struct CUevent_st *Event;
typedef struct CUstream_st *Stream;
enum Api { CREATE, RECORD, SYNCHRONIZE, DESTROY };
static const char *const names[] = {
    "cudaEventCreateWithFlags", "cudaEventRecord", "cudaEventSynchronize",
    "cudaEventDestroy"};
static int (*real_create)(Event *, unsigned int);
static int (*real_record)(Event, Stream);
static int (*real_synchronize)(Event);
static int (*real_destroy)(Event);
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t table_mutex = PTHREAD_MUTEX_INITIALIZER;

struct EventSlot { Event event; unsigned int flags; int occupied; };
struct Row {
  long tid;
  unsigned int flags;
  int occupied, api, known, result, timing_valid;
  uint64_t calls, wall_ns, thread_cpu_ns;
};
static struct EventSlot events[C8_EVENT_TIMER_CAPACITY];
static struct Row rows[C8_EVENT_TIMER_ROWS];
static uint64_t event_capacity_misses, row_capacity_misses, unknown_event_calls;
static uint64_t invalid_timing_calls, observed_calls;
static int counter_overflow;

static void emit(const char *buffer, size_t count) {
  int saved_errno = errno;
  while (count) {
    ssize_t done = write(STDERR_FILENO, buffer, count);
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) break;  /* Logging cannot change the forwarded CUDA result. */
    buffer += done;
    count -= (size_t)done;
  }
  errno = saved_errno;
}

static void missing_symbol(const char *symbol) {
  char line[256];
  int n = snprintf(line, sizeof(line),
      "{\"kind\":\"C8_CUDA_EVENT_TIMER_ERROR\",\"missing_symbol\":\"%s\",\"exit_code\":127}\n",
      symbol);
  if (n > 0 && (size_t)n < sizeof(line)) emit(line, (size_t)n);
  _exit(127);
}

static void resolve_symbols(void) {
  void *symbol;
#define RESOLVE(target, name) do { \
    symbol = dlsym(RTLD_NEXT, name); \
    if (!symbol || sizeof(target) != sizeof(symbol)) missing_symbol(name); \
    memcpy(&(target), &symbol, sizeof(target)); \
  } while (0)
  RESOLVE(real_create, "cudaEventCreateWithFlags");
  RESOLVE(real_record, "cudaEventRecord");
  RESOLVE(real_synchronize, "cudaEventSynchronize");
  RESOLVE(real_destroy, "cudaEventDestroy");
#undef RESOLVE
}

static void add(uint64_t *value, uint64_t amount) {
  if (UINT64_MAX - *value < amount) {
    *value = UINT64_MAX;
    counter_overflow = 1;
  } else *value += amount;
}

static int lookup(Event event, unsigned int *flags) {
  int found = 0;
  pthread_mutex_lock(&table_mutex);
  size_t first = ((uintptr_t)event >> 3) % C8_EVENT_TIMER_CAPACITY;
  for (size_t probe = 0; probe < C8_EVENT_TIMER_CAPACITY; ++probe) {
    size_t i = (first + probe) % C8_EVENT_TIMER_CAPACITY;
    if (!events[i].occupied) break;
    if (events[i].occupied == 1 && events[i].event == event) {
      *flags = events[i].flags;
      found = 1;
      break;
    }
  }
  pthread_mutex_unlock(&table_mutex);
  return found;
}

static void created(Event event, unsigned int flags) {
  pthread_mutex_lock(&table_mutex);
  size_t empty = C8_EVENT_TIMER_CAPACITY;
  size_t first = ((uintptr_t)event >> 3) % C8_EVENT_TIMER_CAPACITY;
  for (size_t probe = 0; probe < C8_EVENT_TIMER_CAPACITY; ++probe) {
    size_t i = (first + probe) % C8_EVENT_TIMER_CAPACITY;
    if (events[i].occupied == 1 && events[i].event == event) {
      events[i].flags = flags;
      pthread_mutex_unlock(&table_mutex);
      return;
    }
    if (events[i].occupied != 1 && empty == C8_EVENT_TIMER_CAPACITY) empty = i;
    if (!events[i].occupied) break;
  }
  if (empty == C8_EVENT_TIMER_CAPACITY) add(&event_capacity_misses, 1);
  else events[empty] = (struct EventSlot){event, flags, 1};
  pthread_mutex_unlock(&table_mutex);
}

static void destroyed(Event event) {
  pthread_mutex_lock(&table_mutex);
  size_t first = ((uintptr_t)event >> 3) % C8_EVENT_TIMER_CAPACITY;
  for (size_t probe = 0; probe < C8_EVENT_TIMER_CAPACITY; ++probe) {
    size_t i = (first + probe) % C8_EVENT_TIMER_CAPACITY;
    if (!events[i].occupied) break;
    if (events[i].occupied == 1 && events[i].event == event) {
      events[i].occupied = 2; /* Tombstone: preserve later hash-collision chains. */
      break;
    }
  }
  pthread_mutex_unlock(&table_mutex);
}

struct Stamp { struct timespec wall, cpu; int valid; };
static struct Stamp stamp(void) {
  struct Stamp value;
  int wall = clock_gettime(CLOCK_MONOTONIC, &value.wall);
  int cpu = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value.cpu);
  value.valid = wall == 0 && cpu == 0;
  return value;
}
static uint64_t elapsed(struct timespec before, struct timespec after) {
  int64_t ns = (int64_t)(after.tv_sec - before.tv_sec) * 1000000000LL
               + after.tv_nsec - before.tv_nsec;
  return ns < 0 ? 0 : (uint64_t)ns;
}

static void record_call(long tid, enum Api api, unsigned int flags, int known,
                        int result, struct Stamp start, struct Stamp end) {
  int valid = start.valid && end.valid;
  pthread_mutex_lock(&table_mutex);
  add(&observed_calls, 1);
  if (!known) add(&unknown_event_calls, 1);
  if (!valid) add(&invalid_timing_calls, 1);
  struct Row *row = NULL;
  for (size_t i = 0; i < C8_EVENT_TIMER_ROWS; ++i) {
    if (!rows[i].occupied) {
      row = &rows[i]; /* Aggregate rows are append-only, without holes. */
      break;
    }
    if (rows[i].tid == tid && rows[i].api == (int)api &&
        rows[i].known == known && (!known || rows[i].flags == flags) &&
        rows[i].result == result && rows[i].timing_valid == valid) {
      row = &rows[i];
      break;
    }
  }
  if (!row) add(&row_capacity_misses, 1);
  else {
    if (!row->occupied)
      *row = (struct Row){.tid=tid, .flags=flags, .occupied=1, .api=api,
                          .known=known, .result=result, .timing_valid=valid};
    add(&row->calls, 1);
    if (valid) {
      add(&row->wall_ns, elapsed(start.wall, end.wall));
      add(&row->thread_cpu_ns, elapsed(start.cpu, end.cpu));
    }
  }
  pthread_mutex_unlock(&table_mutex);
}

/* Resolve/take the TID before timing, then restore incoming errno before the
 * REAL API. Preserve errno immediately after it, before end clocks/bookkeeping. */
#define PREPARE() \
  int incoming_errno = errno; \
  if (pthread_once(&resolve_once, resolve_symbols)) missing_symbol("pthread_once"); \
  long tid = (long)syscall(SYS_gettid)
#define BEGIN_API() struct Stamp start = stamp(); errno = incoming_errno
#define END_API(api, flags, known) do { \
  int result_errno = errno; \
  struct Stamp end = stamp(); \
  record_call(tid, api, flags, known, result, start, end); \
  errno = result_errno; \
} while (0)

int cudaEventCreateWithFlags(Event *event, unsigned int flags) {
  PREPARE();
  BEGIN_API();
  int result = real_create(event, flags);
  int original_errno = errno;
  struct Stamp end = stamp();
  if (result == 0 && event) created(*event, flags);
  record_call(tid, CREATE, flags, 1, result, start, end);
  errno = original_errno;
  return result;
}

int cudaEventRecord(Event event, Stream stream) {
  PREPARE();
  unsigned int flags = 0;
  int known = lookup(event, &flags);
  BEGIN_API();
  int result = real_record(event, stream);
  END_API(RECORD, flags, known);
  return result;
}

int cudaEventSynchronize(Event event) {
  PREPARE();
  unsigned int flags = 0;
  int known = lookup(event, &flags);
  BEGIN_API();
  int result = real_synchronize(event);
  END_API(SYNCHRONIZE, flags, known);
  return result;
}

int cudaEventDestroy(Event event) {
  PREPARE();
  unsigned int flags = 0;
  int known = lookup(event, &flags);
  BEGIN_API();
  int result = real_destroy(event);
  int original_errno = errno;
  struct Stamp end = stamp();
  if (result == 0) destroyed(event);
  record_call(tid, DESTROY, flags, known, result, start, end);
  errno = original_errno;
  return result;
}

__attribute__((destructor)) static void print_summary(void) {
  int saved_errno = errno;
  pthread_mutex_lock(&table_mutex);
  uint64_t live = 0;
  for (size_t i = 0; i < C8_EVENT_TIMER_CAPACITY; ++i) live += events[i].occupied == 1;
  for (size_t i = 0; i < C8_EVENT_TIMER_ROWS; ++i) if (rows[i].occupied) {
    char flags[32], line[640];
    struct Row *r = &rows[i];
    if (r->known) snprintf(flags, sizeof(flags), "%u", r->flags);
    else snprintf(flags, sizeof(flags), "null");
    int n = snprintf(line, sizeof(line),
        "{\"kind\":\"C8_CUDA_EVENT_TIMER\",\"schema\":1,\"pid\":%ld,\"tid\":%ld,"
        "\"api\":\"%s\",\"flags\":%s,\"retval\":%d,\"calls\":%" PRIu64 ","
        "\"wall_ns\":%" PRIu64 ",\"thread_cpu_ns\":%" PRIu64 ","
        "\"timing_valid\":%s,\"injected_diagnostic\":true}\n",
        (long)getpid(), r->tid, names[r->api], flags, r->result, r->calls, r->wall_ns,
        r->thread_cpu_ns, r->timing_valid ? "true" : "false");
    if (n > 0 && (size_t)n < sizeof(line)) emit(line, (size_t)n);
  }
  char line[512];
  int n = snprintf(line, sizeof(line),
      "{\"kind\":\"C8_CUDA_EVENT_TIMER_SUMMARY\",\"schema\":1,\"pid\":%ld,"
      "\"observed_calls\":%" PRIu64 ","
      "\"event_capacity\":%d,\"row_capacity\":%d,\"tracked_live_events\":%" PRIu64 ","
      "\"event_capacity_misses\":%" PRIu64 ",\"row_capacity_misses\":%" PRIu64 ","
      "\"unknown_event_calls\":%" PRIu64 ",\"invalid_timing_calls\":%" PRIu64 ",\"counter_overflow\":%s,"
      "\"coverage_complete\":%s,\"flush\":\"normal_process_exit\"}\n",
      (long)getpid(), observed_calls, C8_EVENT_TIMER_CAPACITY, C8_EVENT_TIMER_ROWS, live, event_capacity_misses,
      row_capacity_misses, unknown_event_calls, invalid_timing_calls, counter_overflow ? "true" : "false",
      (observed_calls && !event_capacity_misses && !row_capacity_misses && !unknown_event_calls &&
       !invalid_timing_calls && !counter_overflow) ? "true" : "false");
  if (n > 0 && (size_t)n < sizeof(line)) emit(line, (size_t)n);
  pthread_mutex_unlock(&table_mutex);
  errno = saved_errno;
}
