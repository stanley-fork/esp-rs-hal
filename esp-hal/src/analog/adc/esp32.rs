use core::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

use super::{AdcConfig, Attenuation};
use crate::{
    peripherals::{ADC1, ADC2, RTC_IO, SENS},
    private::{self},
};

pub(super) const NUM_ATTENS: usize = 10;

// ADC2 cannot be used with `radio` functionality on `esp32`, this global helps us to track it's
// state to prevent unexpected behaviour
static ADC2_IN_USE: AtomicBool = AtomicBool::new(false);

/// ADC Error
#[derive(Debug)]
pub enum Error {
    /// `ADC2` is used together with `radio`.
    Adc2InUse,
}

#[doc(hidden)]
/// Tries to "claim" `ADC2` peripheral and set its status
pub fn try_claim_adc2(_: private::Internal) -> Result<(), Error> {
    if ADC2_IN_USE.fetch_or(true, Ordering::Relaxed) {
        Err(Error::Adc2InUse)
    } else {
        Ok(())
    }
}

#[doc(hidden)]
/// Resets `ADC2` usage status to `Unused`
pub fn release_adc2(_: private::Internal) {
    ADC2_IN_USE.store(false, Ordering::Relaxed);
}

/// The sampling/readout resolution of the ADC.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[allow(clippy::enum_variant_names, reason = "peripheral is unstable")]
pub enum Resolution {
    /// 9-bit resolution
    Resolution9Bit  = 0b00,
    /// 10-bit resolution
    Resolution10Bit = 0b01,
    /// 11-bit resolution
    Resolution11Bit = 0b10,
    /// 12-bit resolution
    #[default]
    Resolution12Bit = 0b11,
}

#[doc(hidden)]
pub trait RegisterAccess {
    fn set_bit_width(resolution: u8);

    fn set_sample_bit(resolution: u8);

    fn set_attenuation(channel: usize, attenuation: u8);

    fn clear_dig_force();

    fn set_start_force();

    fn set_en_pad_force();

    fn set_en_pad(channel: u8);

    fn clear_start_sar();

    fn set_start_sar();

    fn read_done_sar() -> bool;

    fn read_data_sar() -> u16;

    fn instance_number() -> u8;
}

impl RegisterAccess for ADC1<'_> {
    fn set_bit_width(resolution: u8) {
        SENS::regs()
            .sar_start_force()
            .modify(|_, w| unsafe { w.sar1_bit_width().bits(resolution) });
    }

    fn set_sample_bit(resolution: u8) {
        SENS::regs()
            .sar_read_ctrl()
            .modify(|_, w| unsafe { w.sar1_sample_bit().bits(resolution) });
    }

    fn set_attenuation(channel: usize, attenuation: u8) {
        SENS::regs().sar_atten1().modify(|r, w| {
            let new_value = (r.bits() & !(0b11 << (channel * 2)))
                | (((attenuation & 0b11) as u32) << (channel * 2));

            unsafe { w.sar1_atten().bits(new_value) }
        });
    }

    fn clear_dig_force() {
        SENS::regs()
            .sar_read_ctrl()
            .modify(|_, w| w.sar1_dig_force().clear_bit());
    }

    fn set_start_force() {
        SENS::regs()
            .sar_meas_start1()
            .modify(|_, w| w.meas1_start_force().set_bit());
    }

    fn set_en_pad_force() {
        SENS::regs()
            .sar_meas_start1()
            .modify(|_, w| w.sar1_en_pad_force().set_bit());
    }

    fn set_en_pad(channel: u8) {
        SENS::regs()
            .sar_meas_start1()
            .modify(|_, w| unsafe { w.sar1_en_pad().bits(1 << channel) });
    }

    fn clear_start_sar() {
        SENS::regs()
            .sar_meas_start1()
            .modify(|_, w| w.meas1_start_sar().clear_bit());
    }

    fn set_start_sar() {
        SENS::regs()
            .sar_meas_start1()
            .modify(|_, w| w.meas1_start_sar().set_bit());
    }

    fn read_done_sar() -> bool {
        SENS::regs()
            .sar_meas_start1()
            .read()
            .meas1_done_sar()
            .bit_is_set()
    }

    fn read_data_sar() -> u16 {
        SENS::regs()
            .sar_meas_start1()
            .read()
            .meas1_data_sar()
            .bits()
    }

    fn instance_number() -> u8 {
        1
    }
}

impl RegisterAccess for ADC2<'_> {
    fn set_bit_width(resolution: u8) {
        SENS::regs()
            .sar_start_force()
            .modify(|_, w| unsafe { w.sar2_bit_width().bits(resolution) });
    }

    fn set_sample_bit(resolution: u8) {
        SENS::regs()
            .sar_read_ctrl2()
            .modify(|_, w| unsafe { w.sar2_sample_bit().bits(resolution) });
    }

    fn set_attenuation(channel: usize, attenuation: u8) {
        SENS::regs().sar_atten2().modify(|r, w| {
            let new_value = (r.bits() & !(0b11 << (channel * 2)))
                | (((attenuation & 0b11) as u32) << (channel * 2));

            unsafe { w.sar2_atten().bits(new_value) }
        });
    }

    fn clear_dig_force() {
        SENS::regs()
            .sar_read_ctrl2()
            .modify(|_, w| w.sar2_dig_force().clear_bit());
    }

    fn set_start_force() {
        SENS::regs()
            .sar_meas_start2()
            .modify(|_, w| w.meas2_start_force().set_bit());
    }

    fn set_en_pad_force() {
        SENS::regs()
            .sar_meas_start2()
            .modify(|_, w| w.sar2_en_pad_force().set_bit());
    }

    fn set_en_pad(channel: u8) {
        SENS::regs()
            .sar_meas_start2()
            .modify(|_, w| unsafe { w.sar2_en_pad().bits(1 << channel) });
    }

    fn clear_start_sar() {
        SENS::regs()
            .sar_meas_start2()
            .modify(|_, w| w.meas2_start_sar().clear_bit());
    }

    fn set_start_sar() {
        SENS::regs()
            .sar_meas_start2()
            .modify(|_, w| w.meas2_start_sar().set_bit());
    }

    fn read_done_sar() -> bool {
        SENS::regs()
            .sar_meas_start2()
            .read()
            .meas2_done_sar()
            .bit_is_set()
    }

    fn read_data_sar() -> u16 {
        SENS::regs()
            .sar_meas_start2()
            .read()
            .meas2_data_sar()
            .bits()
    }

    fn instance_number() -> u8 {
        2
    }
}

/// Analog-to-Digital Converter peripheral driver.
pub struct Adc<'d, ADC, Dm: crate::DriverMode> {
    _adc: ADC,
    attenuations: [Option<Attenuation>; NUM_ATTENS],
    active_channel: Option<u8>,
    _phantom: PhantomData<(Dm, &'d mut ())>,
}

impl<'d, ADCI> Adc<'d, ADCI, crate::Blocking>
where
    ADCI: RegisterAccess + 'd,
{
    /// Configure a given ADC instance using the provided configuration, and
    /// initialize the ADC for use
    ///
    /// # Panics
    ///
    /// `ADC2` cannot be used simultaneously with `radio` functionalities, otherwise this function
    /// will panic.
    pub fn new(adc_instance: ADCI, config: AdcConfig<ADCI>) -> Self {
        if ADCI::instance_number() == 2 && try_claim_adc2(private::Internal).is_err() {
            panic!(
                "ADC2 is already in use by Radio. On ESP32, ADC2 cannot be used simultaneously with Wi-Fi or Bluetooth."
            );
        }

        let sensors = SENS::regs();

        // Set reading and sampling resolution
        let resolution: u8 = config.resolution as u8;

        ADCI::set_bit_width(resolution);
        ADCI::set_sample_bit(resolution);

        // Set attenuation for pins
        let attenuations = config.attenuations;

        for (channel, attentuation) in attenuations.iter().enumerate() {
            if let Some(attenuation) = attentuation {
                ADC1::set_attenuation(channel, *attenuation as u8);
            }
        }

        // Set controller to RTC
        ADCI::clear_dig_force();
        ADCI::set_start_force();
        ADCI::set_en_pad_force();
        sensors
            .sar_touch_ctrl1()
            .modify(|_, w| w.xpd_hall_force().set_bit());
        sensors
            .sar_touch_ctrl1()
            .modify(|_, w| w.hall_phase_force().set_bit());

        // Set power to SW power on
        sensors
            .sar_meas_wait2()
            .modify(|_, w| unsafe { w.force_xpd_sar().bits(0b11) });

        // disable AMP
        sensors
            .sar_meas_wait2()
            .modify(|_, w| unsafe { w.force_xpd_amp().bits(0b10) });
        sensors
            .sar_meas_ctrl()
            .modify(|_, w| unsafe { w.amp_rst_fb_fsm().bits(0) });
        sensors
            .sar_meas_ctrl()
            .modify(|_, w| unsafe { w.amp_short_ref_fsm().bits(0) });
        sensors
            .sar_meas_ctrl()
            .modify(|_, w| unsafe { w.amp_short_ref_gnd_fsm().bits(0) });
        sensors
            .sar_meas_wait1()
            .modify(|_, w| unsafe { w.sar_amp_wait1().bits(1) });
        sensors
            .sar_meas_wait1()
            .modify(|_, w| unsafe { w.sar_amp_wait2().bits(1) });
        sensors
            .sar_meas_wait2()
            .modify(|_, w| unsafe { w.sar_amp_wait3().bits(1) });

        // Do *not* invert the output
        // NOTE: This seems backwards, but was verified experimentally.
        sensors
            .sar_read_ctrl2()
            .modify(|_, w| w.sar2_data_inv().set_bit());

        Adc {
            _adc: adc_instance,
            attenuations: config.attenuations,
            active_channel: None,
            _phantom: PhantomData,
        }
    }

    /// Request that the ADC begin a conversion on the specified pin
    ///
    /// This method takes an [AdcPin](super::AdcPin) reference, as it is
    /// expected that the ADC will be able to sample whatever channel
    /// underlies the pin.
    pub fn read_oneshot<PIN>(&mut self, _pin: &mut super::AdcPin<PIN, ADCI>) -> nb::Result<u16, ()>
    where
        PIN: super::AdcChannel,
    {
        if self.attenuations[PIN::CHANNEL as usize].is_none() {
            panic!("Channel {} is not configured reading!", PIN::CHANNEL);
        }

        if let Some(active_channel) = self.active_channel {
            // There is conversion in progress:
            // - if it's for a different channel try again later
            // - if it's for the given channel, go ahead and check progress
            if active_channel != PIN::CHANNEL {
                return Err(nb::Error::WouldBlock);
            }
        } else {
            // If no conversions are in progress, start a new one for given channel
            self.active_channel = Some(PIN::CHANNEL);

            ADCI::set_en_pad(PIN::CHANNEL);

            ADCI::clear_start_sar();
            ADCI::set_start_sar();
        }

        // Wait for ADC to finish conversion
        let conversion_finished = ADCI::read_done_sar();
        if !conversion_finished {
            return Err(nb::Error::WouldBlock);
        }

        // Get converted value
        let converted_value = ADCI::read_data_sar();

        // Mark that no conversions are currently in progress
        self.active_channel = None;

        Ok(converted_value)
    }
}

impl<ADC1> Adc<'_, ADC1, crate::Blocking> {
    /// Enable the Hall sensor
    pub fn enable_hall_sensor() {
        RTC_IO::regs()
            .hall_sens()
            .modify(|_, w| w.xpd_hall().set_bit());
    }

    /// Disable the Hall sensor
    pub fn disable_hall_sensor() {
        RTC_IO::regs()
            .hall_sens()
            .modify(|_, w| w.xpd_hall().clear_bit());
    }
}

mod adc_implementation {
    crate::analog::adc::impl_adc_interface! {
        ADC1 [
            (GPIO36<'_>, 0), // Alt. name: SENSOR_VP
            (GPIO37<'_>, 1), // Alt. name: SENSOR_CAPP
            (GPIO38<'_>, 2), // Alt. name: SENSOR_CAPN
            (GPIO39<'_>, 3), // Alt. name: SENSOR_VN
            (GPIO33<'_>, 4), // Alt. name: 32K_XP
            (GPIO32<'_>, 5), // Alt. name: 32K_XN
            (GPIO34<'_>, 6), // Alt. name: VDET_1
            (GPIO35<'_>, 7), // Alt. name: VDET_2
        ]
    }

    crate::analog::adc::impl_adc_interface! {
        ADC2 [
            (GPIO4<'_>,  0),
            (GPIO0<'_>,  1),
            (GPIO2<'_>,  2),
            (GPIO15<'_>, 3), // Alt. name: MTDO
            (GPIO13<'_>, 4), // Alt. name: MTCK
            (GPIO12<'_>, 5), // Alt. name: MTDI
            (GPIO14<'_>, 6), // Alt. name: MTMS
            (GPIO27<'_>, 7),
            (GPIO25<'_>, 8),
            (GPIO26<'_>, 9),
        ]
    }
}

impl Drop for ADC2<'_> {
    fn drop(&mut self) {
        release_adc2(private::Internal);
    }
}
