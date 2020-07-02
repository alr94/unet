// vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

void make_training_data()
{

  std::string dirname = "MichelEnergyImage";
  
  TFile * train_file = new TFile("MC_Training_Flattened.root", "RECREATE");
  TDirectory * train_dir = train_file -> mkdir("MichelEnergyImage");
  
  TRandom2 * rand = new TRandom2();

  std::cout <<  "Starting file loop" << std::endl;
  std::string basename = "/eos/user/a/areynold/michelunet_data/80cm/testing/MC_Testing_Image_batch";
  
  for (int batch_num = 3; batch_num < 43; batch_num++)
  // for (int batch_num = 3; batch_num < 4; batch_num++)
  {
    
    std::string name = basename;
    if (batch_num < 10) { name.append("0"); }
    name.append(std::to_string(batch_num));
    name.append(".root");
    TFile * input_data = new TFile(name.c_str(), "READ");
    TDirectoryFile * data_base_dir = 
      (TDirectoryFile*) input_data -> Get(dirname.c_str());
    
    TFile * test_file = new TFile("MC_Testing_T0.root", "RECREATE");
    TDirectory * test_dir = test_file -> mkdir("MichelEnergyImage");
    
    int n_train = 0;
    TList * list = data_base_dir -> GetListOfKeys();
    for (int fileindex = 0; fileindex < list -> LastIndex() + 1; fileindex++)
    // for (int fileindex = 0; fileindex < 1; fileindex++)
    {
      
      if (fileindex % 100 == 0)
      {
        std::cout << name << " " << fileindex  << " / " << list -> LastIndex() << std::endl;
      }
      
      TDirectoryFile * data_dir = (TDirectoryFile*) ((TKey*) list -> At(fileindex)) -> ReadObj();
      data_dir -> cd();
      
      TTree * tree = (TTree*) data_dir -> Get("param tree");
      TTreeReader reader(tree);
      TTreeReaderValue<bool> hast0(reader, "HasT0");
      TTreeReaderValue<double> calibFrac(reader, "CalibFrac");
      TTreeReaderValue<double> trueIonE(reader, "totalTrueIonE");
      
      bool save_dir = true; bool test_save = false;
      while (reader.Next())
      {
        if (*hast0 == true)  
        { 
          save_dir = false; 
          test_save = true;
        }
        if (*calibFrac < 0.5) 
        { 
          save_dir = false; 
          test_save = false;
        }
        
        double save_prob = 0.; 
        if (*trueIonE < 10.)      { save_prob = 91./320.;}
        else if (*trueIonE < 20.) { save_prob = 91./936.; }
        else if (*trueIonE < 30.) { save_prob = 91./999; }
        else if (*trueIonE < 40.) { save_prob = 91./677.; }
        else if (*trueIonE < 50.) { save_prob = 1.; }
        
        double rand_draw = rand -> Uniform(1.);
        if (rand_draw > save_prob) { save_dir = false; }

      }
      
      if (save_dir)
      {
        
        TDirectory * savedir = train_dir -> mkdir(data_dir -> GetTitle());
        if (savedir == nullptr) { continue; }
        savedir -> cd();
        
        TList * keys = data_dir -> GetListOfKeys();
        for (int keyindex = 0; keyindex < keys -> LastIndex() + 1; keyindex++)
        {
          TKey * key = (TKey *) keys -> At(keyindex);
          const char * classname = key -> GetClassName();
          TClass * cl = gROOT -> GetClass(classname);
          if (!cl) { continue; }
          else if (cl -> InheritsFrom(TTree::Class()))
          {
            data_dir -> cd();
            TTree * t = (TTree*)  data_dir -> Get(key -> GetName());
            savedir -> cd();
            TTree * nt = t -> CloneTree(-1, "fast");
            nt -> Write();
          }
          else
          {
            data_dir -> cd();
            TObject * obj = key -> ReadObj();
            savedir -> cd();
            obj -> Write();
            delete obj;
          }
            
        }
        
        n_train += 1;
        
      }
      else if (test_save)
      {
        
        TDirectory * savedir = test_dir -> mkdir(data_dir -> GetTitle());
        if (savedir == nullptr) { continue; }
        savedir -> cd();
        
        TList * keys = data_dir -> GetListOfKeys();
        for (int keyindex = 0; keyindex < keys -> LastIndex() + 1; keyindex++)
        {
          TKey * key = (TKey *) keys -> At(keyindex);
          const char * classname = key -> GetClassName();
          TClass * cl = gROOT -> GetClass(classname);
          if (!cl) { continue; }
          else if (cl -> InheritsFrom(TTree::Class()))
          {
            data_dir -> cd();
            TTree * t = (TTree*)  data_dir -> Get(key -> GetName());
            savedir -> cd();
            TTree * nt = t -> CloneTree(-1, "fast");
            nt -> Write();
          }
          else
          {
            data_dir -> cd();
            TObject * obj = key -> ReadObj();
            savedir -> cd();
            obj -> Write();
            delete obj;
          }
            
        }
        
        n_train += 1;
        
      }
      
      data_dir -> Close(); 
      
      if (n_train > 50000) { break; }
      
    }
    
    test_dir -> Close();
    
    if (n_train > 50000) { break; }
    
  }
    
}
