import wx
import os
import time
import subprocess

class Mywin(wx.Frame): 
   def __init__(self, parent, title): 
      super(Mywin, self).__init__(parent, title = title,size = (600,400))
      panel = wx.Panel(self) 
      vbox = wx.BoxSizer() 
      self.SetBackgroundColour((51, 153, 255, 150))
         
      self.btn = wx.Button(panel,-1," Dự báo ", pos =(250, 20), size=(100, 50))
      self.btn.SetBackgroundColour((150, 255, 153,200))
      # self.btn.SetSize((200, 100))
      # vbox.Add(self.btn) 
      self.btn.Bind(wx.EVT_BUTTON,self.OnClicked) 

      wx.SizerFlags.DisableConsistencyChecks() 
      self.tbtn = wx.ToggleButton(panel , -1, "Kích hoạt dự báo tự động", size=(200, 50), pos=(200, 100))
      self.tbtn.SetBackgroundColour((255, 0, 102, 200))
      # vbox.Add(self.tbtn)
      self.tbtn.Bind(wx.EVT_TOGGLEBUTTON,self.OnToggle)

      self.btn = wx.Button(panel, -1, "Nhập dữ liệu để dự đoán",size=(160, 50), pos=(220, 180))
      self.btn.SetBackgroundColour((255, 153, 0, 200))
      # vbox.Add(self.btn)
      self.btn.Bind(wx.EVT_BUTTON, self.OnClicked_2)

      hbox = wx.BoxSizer(wx.HORIZONTAL)
         

         
      vbox.Add(hbox,1,wx.ALIGN_CENTER) 
      panel.SetSizer(vbox) 
        
      self.Centre() 
      self.Show() 
      self.Fit()  
		
   def OnClicked(self, event):
        btn = event.GetEventObject().GetLabel()
        subprocess.run(['python', '-m', 'Main_Forecast.py'], shell=True)
        print("Dự báo đã hoàn thành bằng Button =", btn)

   def OnClicked_2(self, event):
        btn = event.GetEventObject().GetLabel()
        subprocess.run(['python', '-m', 'Main_Prediction.py'], shell=True)
        print("Dự đoán kết thúc bởi Button =", btn)

   def OnToggle(self, event):
        state = event.GetEventObject().GetValue()

        if state == True:
            event.GetEventObject().SetLabel("Tắt tính năng tự động dự đoán")
            while state == True:
                print("Đã bật Tự động dự đoán (cập nhật hàng ngày)")
                subprocess.run(['python', '-m', 'Main_Forecast.py'], shell=True)
                print("Đã hoàn tất cập nhật hàng ngày. Đợi ngày hôm sau.")
                i = 1
                while i < 20:
                    print("Vui lòng đợi " + str(20 - i) + " giây")
                    time.sleep(1)
                    i += 1
        else:
            print("Tự động dự đoán bị vô hiệu hóa")
            event.GetEventObject().SetLabel("Kích hoạt tính năng tự động dự đoán")

app = wx.App()
Mywin(None, 'Dự đoán thời tiết bằng Machine Learning')
app.MainLoop()