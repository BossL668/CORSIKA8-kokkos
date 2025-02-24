#pragma once

// common blocks as
  // defined in epos.inc
extern "C" {
   struct CICNT {
    int inicnt;
};

struct HADR6 {
    int intpol;
    int isigma;
    int iomega;
    int isetcs;
};

  struct NUCL6 {
    int infragm;
  };

  struct CJINTI {
    int iorsce;
    int iorsdf;
    int iorshh;
    int ionudi;
  } ;

  struct NXSAIR {
    float airznxs[3];
    float airanxs[3];
    float airwnxs[3];
    float airavznxs;
    float airavanxs;
  };

   struct APPLI {
    int iappl;
    int model;
  };

   struct XSAPPLI {
    int iapplxs;
    int modelxs;
  };

   struct EVENTS {
    int nevent;
    int nfull;
    int nfreeze;
    int ninicon;
  };

   struct XSEVENT {
    int neventxs;
    int iframexs;
  };

  //   common/metr1/iospec,iocova,iopair,iozero,ioflac,iomom
   struct METR1 {
    int iospec;
    int iocova;
    int iopair;
    int iozero;
    int ioflac;
    int iomom;
  } ;

   struct OTHE2 {
    int ifrade;
    int iframe;
    int idecay;
    int jdecay;
    int iremn;
  } ;

   struct METR7 {
    int ktnbod;
  } ;

struct HAD12 {
    float egylow;
    float egyfac;
  } ;

  struct NUCL1 {
    int laproj;
    int maproj;
    int latarg;
    int matarg;
    float core;
    float fctrmx;
  };

   struct CHADRON {
    float amproj;
    float amtarg;
    float ypjtl;
    float yhaha;
    float pnullx;
  } ;

   struct HADR2 {
    int iomodl;
    int idproj;
    int idtarg;
    float wexcit;
  } ;

  struct HADR25 {
    int idprojin;
    int idtargin;
    float rexdifi[4];
    float rexndii[4];
    int irdmpr;
    int isoproj;
    int isotarg;
  } ;

   struct LEPT1 {
    float engy;
    float elepti;
    float elepto;
    float angmue;
    int icinpu;
  };

struct ENRGY {
    float egymin;
    float egymax;
    float elab;
    float ecms;
    float ekin;
  };

struct HADR1 {
    float pnll;
    float ptq;
    float exmass;
    float cutmss;
    float wproj;
    float wtarg;
  } ;

  inline unsigned int constexpr idxD0 = 0;
  inline unsigned int constexpr idxD1 = 2;
  inline unsigned int constexpr idxD = 1;
  inline unsigned int constexpr nclha = 4;
  inline unsigned int constexpr nclegy = 100;

  struct DPARAM {
    float alpD[nclha][nclha][idxD1 - idxD0 + 1];
    float alpdp[nclha][nclha][idxD1 - idxD0 + 1];
    float alpDpp[nclha][nclha][idxD1 - idxD0 + 1];
    float betD[nclha][nclha][idxD1 - idxD0 + 1];
    float betDp[nclha][nclha][idxD1 - idxD0 + 1];
    float betDpp[nclha][nclha][idxD1 - idxD0 + 1];
    float gamD[nclha][nclha][idxD1 - idxD0 + 1];
    float delD[nclha][nclha][idxD1 - idxD0 + 1];
    int idxDmin;
    float bmxdif[nclha][nclha];
    float bkmxndif;
  } ;

   struct CEVT {
    float phievt;
    int nevt;
    float bimevt;
    int kolevt;
    int koievt;
    float pmxevt;
    float egyevt;
    int npjevt;
    int ntgevt;
    int npnevt;
    int nppevt;
    int ntnevt;
    int ntpevt;
    int jpnevt;
    int jppevt;
    int jtnevt;
    int jtpevt;
    float xbjevt;
    float qsqevt;
    int nglevt;
    float zppevt;
    float zptevt;
    int minfra;
    int maxfra;
    int kohevt;
  } ;

   struct CSEED {
    double seedi;
    double seedj;
    double seedj2;
    double seedc;
    int iseqini;
    int iseqsim;
  };

struct OTHE1 {
    int istore;
    int istmax;
    int gaumx;
    int irescl;
    int ntrymx;
    int nclean;
    int iopdg;
    int ioidch;
  };

struct FILES {
    int ifop;
    int ifmt;
    int ifch;
    int ifcx;
    int ifhi;
    int ifdt;
    int ifcp;
    int ifdr;
  };

struct FNAME {
    char fnch[500];
    char fnhi[500];
    char fndt[500];
    char fnii[500];
    char fnid[500];
    char fnie[500];
    char fnrj[500];
    char fnmt[500];
    char fngrv[500];
    char fncp[500];
    char fnnx[500];
    char fncs[500];
    char fndr[500];
    char fnhpf[500];
  } ;

   struct  NFNAME {
    int nfnch;
    int nfnhi;
    int nfndt;
    int nfnii;
    int nfnid;
    int nfnie;
    int nfnrj;
    int nfnmt;
    int nfngrv;
    int nfncp;
    int nfnnx;
    int nfncs;
    int nfndr;
    int nfnhpf;
  } ;

   struct PRNT1 {
    int iprmpt;
    int ish;
    int ishsub;
    int irandm;
    int irewch;
    int iecho;
    int modsho;
    int idensi;
  };

 struct PRNT3 {
    int ishevt;
    int ixtau;
    int iwseed; //! 1: printout seed
    int jwseed;
    int ixgeometry;
  };

namespace EPOS_LHC {
  inline unsigned int constexpr mmry = 1;
  inline unsigned int constexpr mxptl = 200000 / mmry;
  inline unsigned int constexpr mxnody = 200;
  }
  
   struct CPTL {
    int nptl;
    float pptl[EPOS_LHC::mxptl][5];
    int iorptl[EPOS_LHC::mxptl];
    int idptl[EPOS_LHC::mxptl];
    int istptl[EPOS_LHC::mxptl];
    float tivptl[EPOS_LHC::mxptl][2];
    int ifrptl[EPOS_LHC::mxptl][2];
    int jorptl[EPOS_LHC::mxptl];
    float xorptl[EPOS_LHC::mxptl][4];
    int ibptl[EPOS_LHC::mxptl][4];
    int ityptl[EPOS_LHC::mxptl];
  } ;

   struct  HADR5 {
    float sigtot;
    float sigcut;
    float sigela;
    float sloela;
    float sigsd;
    float sigine;
    float sigdif;
    float sigineaa;
    float sigtotaa;
    float sigelaaa;
    float sigcutaa;
    float sigdd;
  };

  struct NODCY {
    int nrnody;
    int nody[EPOS_LHC::mxnody];
  };

  struct HAD10 {
    int iclpro;
    int icltar;
    int iclegy;
  };
}
