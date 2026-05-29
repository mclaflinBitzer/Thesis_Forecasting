[StagingDefinition = [Kind = "FastCopy"]]
section Section1;
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Construction_Sectors_Predictive_GlobalData xlsx_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Construction_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Construction_Sectors_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Removed top rows" = Table.Skip(Navigation, 16),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}}),
  #"Renamed columns" = Table.RenameColumns(#"Changed column type", {{"Column1", "Country"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Removed top rows 1", "Indicator", each "Construction_Sector"), {{"Indicator", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Reordered columns", {"Country", "Indicator"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", type date}, {"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 1", {{"Year", "Date"}}),
  #"Appended query" = Table.Combine({#"Renamed columns 2", #"Construction_GD-World"})
in
  #"Appended query";
shared #"Construction_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Construction_Sectors_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Removed top rows" = Table.Skip(Navigation, 16),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}}),
  #"Renamed columns" = Table.RenameColumns(#"Changed column type", {{"Column1", "Country"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Removed top rows 1", "Indicator", each "Construction_Sector"), {{"Indicator", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Reordered columns", {"Country", "Indicator"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", type date}, {"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 1", {{"Year", "Date"}}),
  #"Grouped rows" = Table.Group(#"Renamed columns 2", {"Indicator", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom 1" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns 1" = Table.ReorderColumns(#"Added custom 1", {"Country", "Indicator", "Date", "Value"})
in
  #"Reordered columns 1";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Project_GD_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Project_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Global_Data_Project_Data.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Project Value Spread_GD_Sectors", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}, {"Hi Matthew, please take the indicators on sector level; here we have the different indicators ""Leisure & Hospitality"", ""Data Center"" and ""Food Processing Plants"".", type text}, {"Column4", type number}, {"Column5", type number}, {"Column6", type number}, {"Column7", type number}, {"Column8", type number}, {"Column9", type number}, {"Column10", type number}, {"Column11", type number}, {"Column12", type number}, {"Column13", type number}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 4),
  #"Promoted headers 1" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Promoted headers 1", {{"2020", type number}, {"2021", type number}}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Changed column type 1", {"Country", "Sector"}, "Attribute", "Value"),
  #"Renamed columns" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}, {"Sector", "Indicator"}}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Renamed columns", {{"Year", type date}}),
  #"Renamed columns 1" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Appended query" = Table.Combine({#"Renamed columns 1", #"Project_GD-World"})
in
  #"Appended query";
shared #"Project_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Global_Data_Project_Data.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Project Value Spread_GD_Sectors", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}, {"Hi Matthew, please take the indicators on sector level; here we have the different indicators ""Leisure & Hospitality"", ""Data Center"" and ""Food Processing Plants"".", type text}, {"Column4", type number}, {"Column5", type number}, {"Column6", type number}, {"Column7", type number}, {"Column8", type number}, {"Column9", type number}, {"Column10", type number}, {"Column11", type number}, {"Column12", type number}, {"Column13", type number}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 4),
  #"Removed blank rows" = Table.SelectRows(#"Removed top rows", each not List.IsEmpty(List.RemoveMatchingItems(Record.FieldValues(_), {"", null}))),
  #"Promoted headers 1" = Table.PromoteHeaders(#"Removed blank rows", [PromoteAllScalars = true]),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Promoted headers 1", {{"2020", type number}, {"2021", type number}}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Changed column type 1", {"Country", "Sector"}, "Attribute", "Value"),
  #"Renamed columns" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}, {"Sector", "Indicator"}}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Renamed columns", {{"Year", type date}}),
  #"Renamed columns 1" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Grouped rows" = Table.Group(#"Renamed columns 1", {"Indicator", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Date", "Value"})
in
  #"Reordered columns";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Macroeconomic_Economy_Predictive_GlobalData xlsx_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Macro_Econ_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Economy_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 15),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Currency"},{"Column4","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Removed top rows 1", {"Country", "Indicator", "Currency", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Replaced value" = Table.ReplaceValue(#"Renamed columns 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Replaced value", {{"Year", type date}, {"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 1", {{"Year", "Date"}}),
  #"Appended query" = Table.Combine({#"Renamed columns 2", #"Macro_Econ_GD-World"})
in
  #"Appended query";
shared #"Macro_Econ_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Economy_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 15),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Currency"},{"Column4","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Removed top rows 1", {"Country", "Indicator", "Currency", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Replaced value" = Table.ReplaceValue(#"Renamed columns 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Replaced value", {{"Year", type date}, {"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 1", {{"Year", "Date"}}),
  #"Grouped rows" = Table.Group(#"Renamed columns 2", {"Indicator", "Currency", "Unit", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Currency", "Unit", "Date", "Value"})
in
  #"Reordered columns";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Macroeconomic_Sectors_Predictive_GlobalData xlsx_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Macro_Sector_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Sectors_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 14),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Filled down" = Table.FillDown(#"Removed top rows 1", {"Country"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Filled down", {"Country", "Indicator", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", Int64.Type}}),
  #"Replaced value" = Table.ReplaceValue(#"Changed column type 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Replaced value", {{"Value", type number}, {"Year", type date}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Appended query" = Table.Combine({#"Renamed columns 2", #"Macro_Sector_GD-World"})
in
  #"Appended query";
shared #"Macro_Sector_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Sectors_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 14),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Filled down" = Table.FillDown(#"Removed top rows 1", {"Country"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Filled down", {"Country", "Indicator", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", Int64.Type}}),
  #"Replaced value" = Table.ReplaceValue(#"Changed column type 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Replaced value", {{"Value", type number}, {"Year", type date}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Grouped rows" = Table.Group(#"Renamed columns 2", {"Indicator", "Unit", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Unit", "Date", "Value"})
in
  #"Reordered columns";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Macroeconomic_Trade_Investment_Predictive_GlobalData xlsx_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Macro_Trade_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Trade_Investment_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 14),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Currency"},{"Column4","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Filled down" = Table.FillDown(#"Removed top rows 1", {"Country"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Filled down", {"Country", "Indicator", "Currency", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", type date}}),
  #"Replaced value" = Table.ReplaceValue(#"Changed column type 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Replaced value", {{"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Appended query" = Table.Combine({#"Renamed columns 2", #"Macro_Trade_GD-World"})
in
  #"Appended query";
shared #"Macro_Trade_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Macroeconomic_Trade_Investment_Predictive_GlobalData.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Report", Kind = "Sheet"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(Navigation, {{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 14),
  #"Promoted headers" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Renamed columns" = Table.RenameColumns(#"Promoted headers", {{"Column1", "Country"},{"Column2","Indicator"},{"Column3","Currency"},{"Column4","Unit"}}),
  #"Removed top rows 1" = Table.Skip(#"Renamed columns", 1),
  #"Filled down" = Table.FillDown(#"Removed top rows 1", {"Country"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Filled down", {"Country", "Indicator", "Currency", "Unit"}, "Attribute", "Value"),
  #"Renamed columns 1" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Year"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns 1", {{"Year", type date}}),
  #"Replaced value" = Table.ReplaceValue(#"Changed column type 1", "-", null, Replacer.ReplaceValue, {"Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Replaced value", {{"Value", type number}}),
  #"Renamed columns 2" = Table.RenameColumns(#"Changed column type 2", {{"Year", "Date"}}),
  #"Grouped rows" = Table.Group(#"Renamed columns 2", {"Indicator", "Currency", "Unit", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Currency", "Unit", "Date", "Value"})
in
  #"Reordered columns";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Oxford_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared Oxford = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Oxecon download - 9 April 2026 10_45_44.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Default", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Replace Null" = Table.TransformColumns(
    #"Promoted headers",
    {
    {"2010Q1", each if _ = "NA" then null else _, type text},
    {"2010Q2", each if _ = "NA" then null else _, type text},
    {"2010Q3", each if _ = "NA" then null else _, type text},
    {"2010Q4", each if _ = "NA" then null else _, type text},

    {"2011Q1", each if _ = "NA" then null else _, type text},
    {"2011Q2", each if _ = "NA" then null else _, type text},
    {"2011Q3", each if _ = "NA" then null else _, type text},
    {"2011Q4", each if _ = "NA" then null else _, type text},

    {"2012Q1", each if _ = "NA" then null else _, type text},
    {"2012Q2", each if _ = "NA" then null else _, type text},
    {"2012Q3", each if _ = "NA" then null else _, type text},
    {"2012Q4", each if _ = "NA" then null else _, type text},

    {"2013Q1", each if _ = "NA" then null else _, type text},
    {"2013Q2", each if _ = "NA" then null else _, type text},
    {"2013Q3", each if _ = "NA" then null else _, type text},
    {"2013Q4", each if _ = "NA" then null else _, type text},

    {"2014Q1", each if _ = "NA" then null else _, type text},
    {"2014Q2", each if _ = "NA" then null else _, type text},
    {"2014Q3", each if _ = "NA" then null else _, type text},
    {"2014Q4", each if _ = "NA" then null else _, type text},

    {"2015Q1", each if _ = "NA" then null else _, type text},
    {"2015Q2", each if _ = "NA" then null else _, type text},
    {"2015Q3", each if _ = "NA" then null else _, type text},
    {"2015Q4", each if _ = "NA" then null else _, type text},

    {"2016Q1", each if _ = "NA" then null else _, type text},
    {"2016Q2", each if _ = "NA" then null else _, type text},
    {"2016Q3", each if _ = "NA" then null else _, type text},
    {"2016Q4", each if _ = "NA" then null else _, type text},

    {"2017Q1", each if _ = "NA" then null else _, type text},
    {"2017Q2", each if _ = "NA" then null else _, type text},
    {"2017Q3", each if _ = "NA" then null else _, type text},
    {"2017Q4", each if _ = "NA" then null else _, type text},

    {"2018Q1", each if _ = "NA" then null else _, type text},
    {"2018Q2", each if _ = "NA" then null else _, type text},
    {"2018Q3", each if _ = "NA" then null else _, type text},
    {"2018Q4", each if _ = "NA" then null else _, type text},

    {"2019Q1", each if _ = "NA" then null else _, type text},
    {"2019Q2", each if _ = "NA" then null else _, type text},
    {"2019Q3", each if _ = "NA" then null else _, type text},
    {"2019Q4", each if _ = "NA" then null else _, type text},

    {"2020Q1", each if _ = "NA" then null else _, type text},
    {"2020Q2", each if _ = "NA" then null else _, type text},
    {"2020Q3", each if _ = "NA" then null else _, type text},
    {"2020Q4", each if _ = "NA" then null else _, type text},

    {"2021Q1", each if _ = "NA" then null else _, type text},
    {"2021Q2", each if _ = "NA" then null else _, type text},
    {"2021Q3", each if _ = "NA" then null else _, type text},
    {"2021Q4", each if _ = "NA" then null else _, type text},

    {"2022Q1", each if _ = "NA" then null else _, type text},
    {"2022Q2", each if _ = "NA" then null else _, type text},
    {"2022Q3", each if _ = "NA" then null else _, type text},
    {"2022Q4", each if _ = "NA" then null else _, type text},

    {"2023Q1", each if _ = "NA" then null else _, type text},
    {"2023Q2", each if _ = "NA" then null else _, type text},
    {"2023Q3", each if _ = "NA" then null else _, type text},
    {"2023Q4", each if _ = "NA" then null else _, type text},

    {"2024Q1", each if _ = "NA" then null else _, type text},
    {"2024Q2", each if _ = "NA" then null else _, type text},
    {"2024Q3", each if _ = "NA" then null else _, type text},
    {"2024Q4", each if _ = "NA" then null else _, type text},

    {"2025Q1", each if _ = "NA" then null else _, type text},
    {"2025Q2", each if _ = "NA" then null else _, type text},
    {"2025Q3", each if _ = "NA" then null else _, type text},
    {"2025Q4", each if _ = "NA" then null else _, type text},

    {"2026Q1", each if _ = "NA" then null else _, type text},
    {"2026Q2", each if _ = "NA" then null else _, type text},
    {"2026Q3", each if _ = "NA" then null else _, type text},
    {"2026Q4", each if _ = "NA" then null else _, type text},

    {"2027Q1", each if _ = "NA" then null else _, type text},
    {"2027Q2", each if _ = "NA" then null else _, type text},
    {"2027Q3", each if _ = "NA" then null else _, type text},
    {"2027Q4", each if _ = "NA" then null else _, type text},

    {"2028Q1", each if _ = "NA" then null else _, type text},
    {"2028Q2", each if _ = "NA" then null else _, type text},
    {"2028Q3", each if _ = "NA" then null else _, type text},
    {"2028Q4", each if _ = "NA" then null else _, type text},

    {"2029Q1", each if _ = "NA" then null else _, type text},
    {"2029Q2", each if _ = "NA" then null else _, type text},
    {"2029Q3", each if _ = "NA" then null else _, type text},
    {"2029Q4", each if _ = "NA" then null else _, type text},

    {"2030Q1", each if _ = "NA" then null else _, type text},
    {"2030Q2", each if _ = "NA" then null else _, type text}
}
),
  #"Changed column type" = Table.TransformColumnTypes(#"Replace Null", {{"Location", type text}, {"Indicator", type text}, {"Sector", type text}, {"Units", type text}, {"Scale", type text}, {"Measurement", type text}, {"2015Q1", type number}, {"2015Q2", type number}, {"2015Q3", type number}, {"2015Q4", type number}, {"2016Q1", type number}, {"2016Q2", type number}, {"2016Q3", type number}, {"2016Q4", type number}, {"2017Q1", type number}, {"2017Q2", type number}, {"2017Q3", type number}, {"2017Q4", type number}, {"2018Q1", type number}, {"2018Q2", type number}, {"2018Q3", type number}, {"2018Q4", type number}, {"2019Q1", type number}, {"2019Q2", type number}, {"2019Q3", type number}, {"2019Q4", type number}, {"2020Q1", type number}, {"2020Q2", type number}, {"2020Q3", type number}, {"2020Q4", type number}, {"2021Q1", type number}, {"2021Q2", type number}, {"2021Q3", type number}, {"2021Q4", type number}, {"2022Q1", type number}, {"2022Q2", type number}, {"2022Q3", type number}, {"2022Q4", type number}, {"2023Q1", type number}, {"2023Q2", type number}, {"2023Q3", type number}, {"2023Q4", type number}, {"2024Q1", type number}, {"2024Q2", type number}, {"2024Q3", type number}, {"2024Q4", type number}, {"2025Q1", type number}, {"2025Q2", type number}, {"2025Q3", type number}, {"2025Q4", type number}, {"2026Q1", type number}, {"2026Q2", type number}, {"2026Q3", type number}, {"2026Q4", type number}, {"2027Q1", type number}, {"2027Q2", type number}, {"2027Q3", type number}, {"2027Q4", type number}, {"2028Q1", type number}, {"2028Q2", type number}, {"2028Q3", type number}, {"2028Q4", type number}, {"2029Q1", type number}, {"2029Q2", type number}, {"2029Q3", type number}, {"2029Q4", type number}, {"2030Q1", type number}, {"2030Q2", type number}, {"2030Q3", type number}, {"2030Q4", type number}, {"Source", type text}, {"Seasonally adjusted", type text}, {"Base year price", Int64.Type}, {"Base year index", Int64.Type}, {"Historical end", type text}, {"Date of last update", type date}, {"Source details", type text}, {"Additional source details", type text}, {"Contact email", type text}, {"Location code", type text}, {"Indicator code", type text}}),
  #"Renamed columns" = Table.RenameColumns(#"Changed column type", {{"Location", "Country"}, {"Indicator", "original_indicator"}}),
  #"Added custom" = Table.AddColumn(#"Renamed columns", "Indicator", each [Sector] & "_" & [original_indicator]),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country",  "Units", "Scale", "Indicator", "Measurement",
   "Source", "Seasonally adjusted", "Base year price", "Base year index", "Historical end", "Date of last update", "Source details", "Additional source details", "Contact email", "Location code", "Indicator code","2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2012Q2", "2012Q3", "2012Q4", "2013Q1", "2013Q2", "2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4", "2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4", "2028Q1", "2028Q2", "2028Q3", "2028Q4", "2029Q1", "2029Q2", "2029Q3", "2029Q4", "2030Q1", "2030Q2", "2030Q3", "2030Q4"}),
  #"Removed columns" = Table.RemoveColumns(#"Reordered columns", {"original_indicator", "Sector", "Measurement", "Source",
    "Scale","Units","Seasonally adjusted","Base year price", "Base year index","Historical end", "Date of last update", "Source details", "Additional source details", "Contact email", "Location code"}),
  #"Replaced value" = Table.ReplaceValue(#"Removed columns", 0, null, Replacer.ReplaceValue, {"2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2012Q2", "2012Q3", "2012Q4", "2013Q1", "2013Q2", "2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4", "2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4", "2028Q1", "2028Q2", "2028Q3", "2028Q4", "2029Q1", "2029Q2", "2029Q3", "2029Q4", "2030Q1", "2030Q2", "2030Q3", "2030Q4"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Replaced value", {"Indicator", "Indicator code", "Country"}, "Attribute", "Value"),
  #"Added custom 1" = Table.AddColumn(#"Unpivoted other columns", "Date", each #date(
    Number.FromText(Text.Start([Attribute],4)),
    (Number.FromText(Text.End([Attribute],1))-1)*3+1,
    1
)),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Added custom 1", {{"Date", type date}, {"Value", type number}}),
  #"Removed columns 1" = Table.RemoveColumns(#"Changed column type 1", {"Attribute"}),
  #"Reordered columns 1" = Table.ReorderColumns(#"Removed columns 1", {"Country", "Indicator", "Indicator code", "Date", "Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Reordered columns 1", {{"Indicator", type text}}),
  #"Appended query" = Table.Combine({#"Changed column type 2", #"Oxford-World"})
in
  #"Appended query";
shared #"Construction_Sectors_Predictive_GlobalData xlsx_DataDestination" = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Construction_Sector", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared #"Oxford-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Oxecon download - 9 April 2026 10_45_44.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Default", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Location", type text}, {"Indicator", type text}, {"Sector", type text}, {"Units", type text}, {"Scale", type text}, {"Measurement", type text}, {"2015Q1", type number}, {"2015Q2", type number}, {"2015Q3", type number}, {"2015Q4", type number}, {"2016Q1", type number}, {"2016Q2", type number}, {"2016Q3", type number}, {"2016Q4", type number}, {"2017Q1", type number}, {"2017Q2", type number}, {"2017Q3", type number}, {"2017Q4", type number}, {"2018Q1", type number}, {"2018Q2", type number}, {"2018Q3", type number}, {"2018Q4", type number}, {"2019Q1", type number}, {"2019Q2", type number}, {"2019Q3", type number}, {"2019Q4", type number}, {"2020Q1", type number}, {"2020Q2", type number}, {"2020Q3", type number}, {"2020Q4", type number}, {"2021Q1", type number}, {"2021Q2", type number}, {"2021Q3", type number}, {"2021Q4", type number}, {"2022Q1", type number}, {"2022Q2", type number}, {"2022Q3", type number}, {"2022Q4", type number}, {"2023Q1", type number}, {"2023Q2", type number}, {"2023Q3", type number}, {"2023Q4", type number}, {"2024Q1", type number}, {"2024Q2", type number}, {"2024Q3", type number}, {"2024Q4", type number}, {"2025Q1", type number}, {"2025Q2", type number}, {"2025Q3", type number}, {"2025Q4", type number}, {"2026Q1", type number}, {"2026Q2", type number}, {"2026Q3", type number}, {"2026Q4", type number}, {"2027Q1", type number}, {"2027Q2", type number}, {"2027Q3", type number}, {"2027Q4", type number}, {"2028Q1", type number}, {"2028Q2", type number}, {"2028Q3", type number}, {"2028Q4", type number}, {"2029Q1", type number}, {"2029Q2", type number}, {"2029Q3", type number}, {"2029Q4", type number}, {"2030Q1", type number}, {"2030Q2", type number}, {"2030Q3", type number}, {"2030Q4", type number}, {"Source", type text}, {"Seasonally adjusted", type text}, {"Base year price", Int64.Type}, {"Base year index", Int64.Type}, {"Historical end", type text}, {"Date of last update", type date}, {"Source details", type text}, {"Additional source details", type text}, {"Contact email", type text}, {"Location code", type text}, {"Indicator code", type text}}),
  #"Renamed columns" = Table.RenameColumns(#"Changed column type", {{"Location", "Country"}, {"Indicator", "original_indicator"}}),
  #"Added custom" = Table.AddColumn(#"Renamed columns", "Indicator", each [Sector] & "_" & [original_indicator]),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country",  "Units", "Scale", "Indicator", "Measurement",
   "Source", "Seasonally adjusted", "Base year price", "Base year index", "Historical end", "Date of last update", "Source details", "Additional source details", "Contact email", "Location code", "Indicator code","2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2012Q2", "2012Q3", "2012Q4", "2013Q1", "2013Q2", "2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4", "2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4", "2028Q1", "2028Q2", "2028Q3", "2028Q4", "2029Q1", "2029Q2", "2029Q3", "2029Q4", "2030Q1", "2030Q2", "2030Q3", "2030Q4"}),
  #"Removed columns" = Table.RemoveColumns(#"Reordered columns", {"original_indicator", "Sector", "Measurement", "Source",
    "Scale","Units","Seasonally adjusted","Base year price", "Base year index","Historical end", "Date of last update", "Source details", "Additional source details", "Contact email", "Location code"}),
  #"Replaced value" = Table.ReplaceValue(#"Removed columns", 0, null, Replacer.ReplaceValue, {"2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2012Q2", "2012Q3", "2012Q4", "2013Q1", "2013Q2", "2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4", "2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4", "2028Q1", "2028Q2", "2028Q3", "2028Q4", "2029Q1", "2029Q2", "2029Q3", "2029Q4", "2030Q1", "2030Q2", "2030Q3", "2030Q4"}),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Replaced value", {"Indicator", "Indicator code", "Country"}, "Attribute", "Value"),
  #"Added custom 1" = Table.AddColumn(#"Unpivoted other columns", "Date", each #date(
    Number.FromText(Text.Start([Attribute],4)),
    (Number.FromText(Text.End([Attribute],1))-1)*3+1,
    1
)),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Added custom 1", {{"Date", type date}, {"Value", type number}}),
  #"Removed columns 1" = Table.RemoveColumns(#"Changed column type 1", {"Attribute"}),
  #"Reordered columns 1" = Table.ReorderColumns(#"Removed columns 1", {"Country", "Indicator", "Indicator code", "Date", "Value"}),
  #"Changed column type 2" = Table.TransformColumnTypes(#"Reordered columns 1", {{"Indicator", type text}}),
  #"Grouped rows" = Table.Group(#"Changed column type 2", {"Indicator", "Indicator code", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom 2" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns 2" = Table.ReorderColumns(#"Added custom 2", {"Country", "Indicator", "Indicator code", "Date", "Value"})
in
  #"Reordered columns 2";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Global_Data_HVAC_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared HVAC_GD = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Global_Data_Project_Data.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Project Value Spread_HVAC", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}, {"Hi Matthew, this is the indicator ""HVAC attribute"" looking into all kinds of project covered on the GlobalData database that are are attributed to HVAC equipment. Therefor please aggregate the sectors into one row per country. Here we are looking into the aggregation of the columns ""sector and sub sector"" ending up in one Indicator that needs to be called ""HVAC Attribute""", type text}, {"Column3", type text}, {"Column4", type number}, {"Column5", type number}, {"Column6", type number}, {"Column7", type number}, {"Column8", type number}, {"Column9", type number}, {"Column10", type number}, {"Column11", type number}, {"Column12", type number}, {"Column13", type number}, {"Column14", type number}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 4),
  #"Promoted headers 1" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Promoted headers 1", {"Country", "Sector", "Sub Sector"}, "Attribute", "Value"),
  #"Renamed columns" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Date"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns", {{"Date", type date}}),
  #"Grouped rows" = Table.Group(#"Changed column type 1", {"Country", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Indicator", each "HVAC"), {{"Indicator", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Date", "Value"}),
  #"Appended query" = Table.Combine({#"Reordered columns", #"HVAC_GD-World"})
in
  #"Appended query";
shared #"Macroeconomic_Economy_Predictive_GlobalData xlsx_DataDestination" = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Macro_Econ", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared #"Macroeconomic_Sectors_Predictive_GlobalData xlsx_DataDestination" = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Macro_Sectors", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared #"Macroeconomic_Trade_Investment_Predictive_GlobalData xlsx_DataDestination" = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Macro_Trade", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared Global_Data_HVAC_DataDestination = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Hvac", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared #"HVAC_GD-World" = let
  Source = Lakehouse.Contents([HierarchicalNavigation = null, EnableVorder = true, OutputMetadataRefresh = true]),
  #"Navigation 1" = Source{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"Navigation 4" = #"Navigation 3"{[Name = "Driver_Data"]}[Content],
  #"Navigation 5" = #"Navigation 4"{[Name = "Global_Data_Project_Data.xlsx"]}[Content],
  #"Imported Excel workbook" = Excel.Workbook(#"Navigation 5", null, true),
  Navigation = #"Imported Excel workbook"{[Item = "Project Value Spread_HVAC", Kind = "Sheet"]}[Data],
  #"Promoted headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"Column1", type text}, {"Hi Matthew, this is the indicator ""HVAC attribute"" looking into all kinds of project covered on the GlobalData database that are are attributed to HVAC equipment. Therefor please aggregate the sectors into one row per country. Here we are looking into the aggregation of the columns ""sector and sub sector"" ending up in one Indicator that needs to be called ""HVAC Attribute""", type text}, {"Column3", type text}, {"Column4", type number}, {"Column5", type number}, {"Column6", type number}, {"Column7", type number}, {"Column8", type number}, {"Column9", type number}, {"Column10", type number}, {"Column11", type number}, {"Column12", type number}, {"Column13", type number}, {"Column14", type number}}),
  #"Removed top rows" = Table.Skip(#"Changed column type", 4),
  #"Promoted headers 1" = Table.PromoteHeaders(#"Removed top rows", [PromoteAllScalars = true]),
  #"Unpivoted other columns" = Table.UnpivotOtherColumns(#"Promoted headers 1", {"Country", "Sector", "Sub Sector"}, "Attribute", "Value"),
  #"Renamed columns" = Table.RenameColumns(#"Unpivoted other columns", {{"Attribute", "Date"}}),
  #"Changed column type 1" = Table.TransformColumnTypes(#"Renamed columns", {{"Date", type date}}),
  #"Grouped rows" = Table.Group(#"Changed column type 1", {"Country", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows", "Indicator", each "HVAC"), {{"Indicator", type text}}),
  #"Reordered columns" = Table.ReorderColumns(#"Added custom", {"Country", "Indicator", "Date", "Value"}),
  #"Grouped rows 1" = Table.Group(#"Reordered columns", {"Indicator", "Date"}, {{"Value", each List.Sum([Value]), type nullable number}}),
  #"Added custom 1" = Table.TransformColumnTypes(Table.AddColumn(#"Grouped rows 1", "Country", each "World"), {{"Country", type text}}),
  #"Reordered columns 1" = Table.ReorderColumns(#"Added custom 1", {"Country", "Indicator", "Date", "Value"})
in
  #"Reordered columns 1";
shared Project_GD_DataDestination = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Sectors", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
shared Oxford_DataDestination = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false, HierarchicalNavigation = true, EnableVorder = true, OutputMetadataRefresh = true]),
  Navigation_1 = Pattern{[workspaceId = "991f5e4b-c174-4ff2-992e-feb17d49d25a"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "22746de3-183e-4327-a844-dceda0b7165c"]}[Data],
  Navigation_3 = Navigation_2{[Id = "bronze", ItemKind = "Schema"]}[Data],
  TableNavigation = Navigation_3{[Id = "DRV_Oxford", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
