package com.fragma.source;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.Font;
import org.apache.poi.ss.usermodel.IndexedColors;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import com.fragma.model.Deliveries;
import com.fragma.model.Matches;
import com.fragma.model.TeamScores;

public class Main {
		
	public static void main(String[] args) {
		
		
		//Creating a workbook for storing the output in a file
		Workbook workbook = null;
		File fileName = new File("F:\\abc.xls");
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(fileName);
		} catch (FileNotFoundException e2) {
		
		System.err.println("File not found");
		}

		workbook = new HSSFWorkbook();

		
		// Create a Sheet
		Sheet sheet = workbook.createSheet("Field first");

		Services service = new Services();
		ArrayList<Matches> listOfMatches = null;
		ArrayList<Deliveries> listOfDeliveries = null;
		//getting list of matches after reading the input csv file
		listOfMatches = service.getMatches();
		//getting list of deliveries after reading the input csv file
		listOfDeliveries = service.getDeliveries();

		// 1
		//finding the top 4 teams who chose to field and writing output to hte file
		service.fieldFirst(listOfMatches, workbook);
		// 2
		sheet = workbook.createSheet("Team Runs detail");
		Font headerFont = workbook.createFont();
		headerFont.setBold(true);
		headerFont.setFontHeightInPoints((short) 14);
		headerFont.setColor(IndexedColors.RED.getIndex());

		// Create a CellStyle with the font
		CellStyle headerCellStyle = workbook.createCellStyle();
		headerCellStyle.setFont(headerFont);
		int excelRowId = 0;
		// Create a Row
		Row headerRow = sheet.createRow(excelRowId++);

		Cell cell = headerRow.createCell(0);
		cell.setCellValue("Year");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(1);
		cell.setCellValue("Team Name");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(2);
		cell.setCellValue("No Of Fours");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(3);
		cell.setCellValue("No Of Sixes");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(4);
		cell.setCellValue("Total Runs");
		cell.setCellStyle(headerCellStyle);
		Row dataRow;

		System.out.println("Year\tTeam Name\t\tNo of Fours\tNo of Sixes\tTotal runs");
		
		/*---------------------------------------------------------------------
	    |
	    |  Purpose:  Getting teams score details 
	    |  parameters:  List of matches and deliveries
	    |  Output: Writing the output to Team Runs detail sheet
	    *-------------------------------------------------------------------*/
		Object[][] teamdetails = service.teamScoreDetails(listOfMatches, listOfDeliveries);
		for (Object[] objects : teamdetails) {
			TeamScores teamScore = (TeamScores) objects[2];
			System.out.println(objects[0].toString() + "\t" + objects[1].toString() + "\t\t" + teamScore.getFours() + "\t"
					+ teamScore.getNoofSixes() + "\t" + teamScore.getTotal());
			dataRow = sheet.createRow(excelRowId++);
			Cell cell1 = dataRow.createCell(0);
			cell1.setCellValue(objects[0].toString());

			cell1 = dataRow.createCell(1);
			cell1.setCellValue(objects[1].toString());

			cell1 = dataRow.createCell(2);
			cell1.setCellValue(teamScore.getFours());
			cell1 = dataRow.createCell(3);
			cell1.setCellValue(teamScore.getNoofSixes());
			cell1 = dataRow.createCell(4);
			cell1.setCellValue(teamScore.getTotal());

		}
		for (int i = 0; i < 5; i++) {
			sheet.autoSizeColumn(i);
		}

		// 3
		/*---------------------------------------------------------------------
	    |
	    |  Purpose:  Getting top 10 best economy bowlers on yearly basis 
	    |  parameters:  List of matches and deliveries
	    |  Output: Writing the output to Best Economy sheet
	    *-------------------------------------------------------------------*/
		Object[][] bestEconomyBowlers = service.bestEconomyBowlers(listOfMatches, listOfDeliveries);
		sheet = workbook.createSheet("Best Economy");
		excelRowId = 0;
		headerRow = sheet.createRow(excelRowId++);
		cell = headerRow.createCell(0);
		cell.setCellValue("Year");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(1);
		cell.setCellValue("Bowler Name");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(2);
		cell.setCellValue("Economy");
		cell.setCellStyle(headerCellStyle);
		System.out.println("Year\tBowler Name\t\tEconomy");
		for (Object[] objects : bestEconomyBowlers) {
			System.out.println(objects[0].toString() + "\t" + objects[1].toString() + "\t\t"
					+ Double.parseDouble(objects[2].toString()));
			dataRow = sheet.createRow(excelRowId++);
			Cell cell1 = dataRow.createCell(0);
			cell1.setCellValue(objects[0].toString());

			cell1 = dataRow.createCell(1);
			cell1.setCellValue(objects[1].toString());

			cell1 = dataRow.createCell(2);
			cell1.setCellValue(Double.parseDouble(objects[2].toString()));
		}
		for (int i = 0; i < 3; i++) {
			sheet.autoSizeColumn(i);
		}

		// 4

		System.out.println("Year\tTeam Name");
		sheet = workbook.createSheet("HighestRunrate");
		excelRowId = 0;
		headerRow = sheet.createRow(excelRowId++);
		cell = headerRow.createCell(0);
		cell.setCellValue("Year");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(1);
		cell.setCellValue("Team Name");
		cell.setCellStyle(headerCellStyle);
		/*---------------------------------------------------------------------
	    |
	    |  Purpose:  Getting team having highest run rate on yearly basis 
	    |  parameters:  List of matches and deliveries
	    |  Output: Writing the output to HighestRunrate sheet
	    *-------------------------------------------------------------------*/
		
		HashMap<Integer, String> highestRunRates = service.getHighestRunRate(listOfMatches, listOfDeliveries);

		for (Entry<Integer, String> entry : highestRunRates.entrySet()) {
			Integer key = entry.getKey();
			String value = entry.getValue();
			System.out.println(key + "\t" + value);
			dataRow = sheet.createRow(excelRowId++);
			Cell cell1 = dataRow.createCell(0);
			cell1.setCellValue(key);
			cell1 = dataRow.createCell(1);
			cell1.setCellValue(value);

		}
		for (int i = 0; i < 2; i++) {
			sheet.autoSizeColumn(i);
		}
		//writing the workbook using file output stream
		try {			
			workbook.write(fos);
			fos.close();
			// Closing the workbook
			workbook.close();
		} catch (FileNotFoundException e) {
			System.err.println("File Not found");			
		} catch (IOException e) {
			System.err.println("Unable to write the file");
			
		}

	}

}
