﻿using BarcodeScanner.Mobile;
using Microsoft.Extensions.Logging;

using Camera.MAUI;
namespace TestScanner
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                  
                  .ConfigureMauiHandlers(handlers =>
                  {
                      // Add the handlers
                      handlers.AddBarcodeScannerHandler();
                  })
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
    		builder.Logging.AddDebug();
           
#endif

            return builder.Build();
        }
    }
}
