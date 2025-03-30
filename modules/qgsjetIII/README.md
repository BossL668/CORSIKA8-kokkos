Note, the "qgsjet-III-04.f" is differnt from the "qgsjet-III-04-org.f"
in the introduction of additional file-handling routines:

```
  void cordataopenfile_(const char* name);
  void cordatafillarray_(double* data, const int& length);
  void cordataclosefile_();
  double cordatanextnumber_();
  void cordatanexttext_(char*, int);
  int cordatacandecompress_();
```

Those functions need to be provided at link time. If provided, this
can be used to read compressed input files (gzip, lzma, etc) as
provided by the corsika-data git project. 

The reference implementation for reading compressed files used
boost-iostreams, but this should be changed since using boost here is
too invasive.
g
