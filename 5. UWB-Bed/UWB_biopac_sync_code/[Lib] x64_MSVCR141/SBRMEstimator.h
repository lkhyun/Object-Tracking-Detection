/****************************************************************************
* Copyright 2018 by Sensor Lab, Seoul R&D Campus Inc.,
* 33 Seongchon-gil, Seocho-gu, Seoul 06765, Korea.
* All rights reserved.
*
* This software is the confidential and proprietary information
* of Samsung Electronics, Inc. ("Confidential Information").  You
* shall not disclose such Confidential Information and shall use
* it only in accordance with the terms of the license agreement
* you entered into with Samsung.
*/

/**
* @file			SBRMEstimator.h
* @brief			header file
* @version		$Revision: 0.1 $
* @date			$Date: 2018-04-19 04:03:28 $
* @author		$Author: sbpark $
*/
#pragma once

#ifndef __SBRMESTIMATOR_H__
#define __SBRMESTIMATOR_H__

/************************************ Include *************************************/
#include "stdio.h"
#include <stdlib.h>
#include <math.h>
#include "SBRMITypes.h"

/************************************ Define *************************************/


/************************************ struct *************************************/
/*************************** Global Function Prototype ***************************/

#ifdef CONFIGURATION_DLL_TYPE
	__declspec(dllimport) XINT8 SBRMEstimator_Initialize(XFLOAT fSamplingRate, XINT32* iOnePacketSize);
#else
	XINT8 SBRMEstimator_Initialize(XFLOAT fSamplingRate, XINT32* iOnePacketSize);
#endif	 //CONFIGURATION_DLL_TYPE

#ifdef CONFIGURATION_DLL_TYPE
	__declspec(dllimport) XBOOL SBRMEstimator_Release();
#else
	XBOOL SBRMEstimator_Release();
#endif // CONFIGURATION_DLL_TYPE

#ifdef CONFIGURATION_DLL_TYPE
	__declspec(dllimport) XINT32 SBRMEstimator_Execute(XFLOAT* fInputData, XINT32* iCurState, XDOUBLE* fEstimatedBR, 
														XINT32* iBRIdx, XFLOAT* fBRSignalVal, XFLOAT* fAbsMovementVal, XFLOAT* pfTmpEnergyProfile);
#else
	XINT32 SBRMEstimator_Execute(XFLOAT* fInputData, XINT32* iCurState, XDOUBLE* fEstimatedBR, 
									XINT32* iBRIdx, XFLOAT* fBRSignalVal, XFLOAT* fAbsMovementVal, XFLOAT* pfTmpEnergyProfile);
#endif // CONFIGURATION_DLL_TYPE

#endif